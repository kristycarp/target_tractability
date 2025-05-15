import requests
from Bio.PDB import PDBParser
from io import StringIO
import xml.etree.ElementTree as ET
import pandas as pd
import math


##################################################
#  FUNCTIONS GENERATING SCORES FROM ENSEMBL IDS  #
##################################################

# does this target have an experimentally-determined structure?
# returns 1 if there is a PDB structure for this target
# returns None if not
def has_pdb_structure(ensembl_id):
    pdb_ids_list = get_pdb_ids(ensembl_id)
    if type(pdb_ids_list) == list:
        if len(pdb_ids_list) > 0:
            return 1
    return None

# does this target have a good computationally-predicted structure?
# returns average confidence of AF2 structure (between 0 and 1)
# 0.9 to 1: very high confidence
# 0.7 to 0.9: high confidence
# 0.5 to 0.7: low confidence
# 0 to 0.5: very low confidence
# returns None if no AF2 structure found or an error occurred in obtaining uniprot id
def has_af2_structure(ensembl_id):
    uniprot_ids = uniprot_from_ensembl(ensembl_id)
    if type(uniprot_ids) == int and uniprot_ids == -1:
        return None

    confidence = None # returns None if no structures found
    for uniprot_id in uniprot_ids:
        url = "https://alphafold.ebi.ac.uk/api/prediction/%s" % uniprot_id
        
        response = requests.get(url)
        if response.status_code != 200:
            continue
    
        api_response = response.json()
        for elem in api_response:
            if not "pdbUrl" in elem.keys():
                continue
            pdb_url = elem["pdbUrl"]
        
            response = requests.get(pdb_url)
            if response.status_code != 200:
                continue
            pdb_io = StringIO(response.text)
            parser = PDBParser(QUIET=True)
            struc = parser.get_structure("example", pdb_io)
            struc_conf = get_structure_confidence(struc)
            struc_conf = struc_conf / 100 # rescale to 0-1 instead of 0-100
            if confidence is None or struc_conf > confidence:
                confidence = struc_conf
    return confidence

# does this target have a structure available of a biologically-relevant ligand interaction?
# returns 1 if there is at least one biolip annotation for this ensembl id
# also returns the list of ligands and whether each ligand is inorganic small molecule / organic small molecule / dna / rna / peptide
# returns None if there are no biolip annotations or an error occurred
# biolip_df parameter is optional and is for supplying locally-downloaded version of biolip df in case the api sends errors for too many queries
def has_biolip_interaction(ensembl_id, biolip_df=None):
    uniprot_ids = uniprot_from_ensembl(ensembl_id)
    if type(uniprot_ids) == int and uniprot_ids == -1:
        return None
    ligands = []
    for uid in uniprot_ids:
        ligands += biolip_query(uid, biolip_df=biolip_df)
    if len(ligands) == 0:
        return None
    else:
        return 1, ligands

# does this target have a good predicted pocket?
# returns the highest p2rank probability score out of all p2rank predicted pockets on all pdb structures for the given ensembl id
# ensures that we only consider chains within each pdb structure that correspond with the desired uniprot id / ensembl id
# if no pockets found or an error occurred, returns None
def has_prank_pdb_pocket(ensembl_id):
    pdbs = get_pdb_ids(ensembl_id)
    uniprots = uniprot_from_ensembl(ensembl_id)

    max_pocket_prob = None
    for pdb_id in pdbs:
        url = "https://prankweb.cz/api/v2/prediction/v4-conservation-hmm/%s/public/prediction.json" % pdb_id
        response = requests.get(url)
        if response.status_code != 200:
            continue
        j = response.json()
        if not "pockets" in j.keys() or len(j["pockets"]) == 0:
            continue
        chain_map = pdb_chain_map(pdb_id)
        relevant_chains = [chain for uniprot_id in chain_map.keys() if uniprot_id in uniprots for chain in chain_map[uniprot_id]]
        pockets = [elem for elem in j["pockets"] if has_relevant_chains(elem, relevant_chains)]
        pocket_probs = [float(pocket["probability"]) for pocket in pockets]
        if len(pocket_probs) == 0:
            continue
        if max_pocket_prob is None:
            max_pocket_prob = max(pocket_probs)
        else:
            max_pocket_prob = max(max_pocket_prob, max(pocket_probs))
    return max_pocket_prob


##################################################
#                HELPER FUNCTIONS                #
##################################################

# returns list of pdb ids for given ensembl id (empty list is possible), or -1 if an error occurred
def get_pdb_ids(ensembl_id):
    api_response = uniprot_query(ensembl_id)

    if type(api_response) == int and api_response == -1 or not "results" in api_response:
        return -1

    pdb_ids = []
    
    for result in api_response["results"]:
        if not "uniProtKBCrossReferences" in result.keys():
            continue
        for elem in result["uniProtKBCrossReferences"]:
            if elem["database"] == "PDB":
                pdb_ids.append(elem["id"])
    return pdb_ids

# for a given PDBParser structure object for an AF2 predicted structure, returns the average per-residue confidence
# cannot use for any structure -- will be meaningless if not structure output from AF2 prediction
def get_structure_confidence(struc):
    model = struc[0]
    atom_confidence = []
    for chain in model:
        for residue in chain:
            for atom in residue:
                atom_confidence.append(atom.get_bfactor())
                break
    return sum(atom_confidence) / len(atom_confidence)

# returns list of uniprot ids linked to a given ensembl id, or -1 if an error occurred
def uniprot_from_ensembl(ensembl_id):
    api_response = uniprot_query(ensembl_id)
    if type(api_response) == int and api_response == -1:
        return -1
    if not "results" in api_response or len(api_response["results"]) == 0:
        return -1
        
    return [elem["primaryAccession"] for elem in api_response["results"]]

# makes a uniprot api query for the given ensembl id and returns the result for downstream use
def uniprot_query(ensembl_id):
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": "(xref:ensembl-%s)" % ensembl_id}
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("error!")
        return -1

    api_response = response.json()
    return api_response

# returns list of ligands and ligand descriptors if there is at least one biolip annotation
# returns empty list if not or if error occurred 
def biolip_query(uniprot_id, biolip_df = None):
    if type(biolip_df) == type(None):
        url = "https://zhanggroup.org/BioLiP/qsearch.cgi"
        params = {"uniprot": uniprot_id,
                  "outfmt": "txt"}
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print("error!")
            return []
    
        response_data = StringIO(response.text)

        cols = ["pdb","chain","resolution","binding site number","ligand id","ligand chain","ligand serial number","binding site residues, pdb numbering","binding site residues, reindexed numbering","catalytic residues, pdb numbering", "catalytic residues, reindexed numbering", "EC number","GO terms","binding affinity, literature","binding affinity, binding moad","binding affinity, pdbbind","binding affinity, bindingdb","uniprot","pubmed","ligand res number","sequence"]
        response_df = pd.read_csv(response_data, sep="\t", names=cols)
    else:
        response_df = biolip_df[biolip_df["uniprot"] == uniprot_id]  

    output = []
    for idx,ligand in enumerate(response_df["ligand id"].unique()):
        if type(ligand) != str:
            print("warning: api query limit likely exceeded, try using local biolip")
            continue
        if ligand in ["dna", "rna", "peptide"]:
            output.append((ligand, ligand))
            continue
        chembl_id = get_chembl_id(ligand)
        if chembl_id == -1:
            output.append((ligand, "small molecule not in chembl"))
            continue
        is_organic = get_organic_def(chembl_id)
        if get_organic_def(chembl_id) == 1:
            output.append((ligand, "organic small molecule"))
        else:
            output.append((ligand, "inorganic small molecule"))
    return output

# for a given biolip / pdb ligand code, returns the corresponding chembl id
# returns -1 if an error occurred or no chembl id found
def get_chembl_id(biolip_lig_code):
    url = "https://zhanglab.comp.nus.edu.sg/BioLiP/sym.cgi"
    params = {"code": biolip_lig_code}
    response = requests.get(url,params=params)
    if response.status_code != 200:
        print("error!")
        return -1

    match = [line for line in response.text.split("\n") if "ChEMBL:" in line]
    if len(match) == 0:
        return -1
    assert len(match) == 1
    match = match[0]
    chembl_id = match.split("</a>")[0]
    chembl_id = chembl_id.split(">")[-1]
    assert chembl_id[:6] == "CHEMBL"
    return chembl_id

# for a given pdb id, returns a dictionary mapping the constituent uniprot ids to their corresponding chains within the structure
def pdb_chain_map(pdb):
    map_dict = dict()
    response = requests.get("https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/%s" % pdb.lower())
    if response.status_code != 200:
        return map_dict
    j = response.json()
    j_uniprot = j[pdb.lower()]["UniProt"]
    for uniprot in j_uniprot:
        map_dict[uniprot] = [elem["chain_id"] for elem in j_uniprot[uniprot]["mappings"]]
    return map_dict

# helper function for has_prank_pdb_pocket
# returns true if the residues in the specified pocket (output from p2rank api) are on any of the specified relevant chains
# returns false otherwise
def has_relevant_chains(pocket, relevant_chains):
    pocket_chains = [elem.split("_")[0] for elem in pocket["residues"]]
    overlap = set(pocket_chains).intersection(set(relevant_chains))
    return len(overlap) > 0

# returns 1 if the provided chembl id is for an organic small molecule
# returns -1 if an error occured, if the chembl id is not for a small moelcule, or if the chembl id is for an inorganic small molecule
def get_organic_def(chembl_id):
    url = "https://www.ebi.ac.uk/chembl/api/data/molecule"
    params = {"molecule_chembl_id__exact": chembl_id}
    response = requests.get(url,params=params)
    if response.status_code != 200:
        print("error!")
        return -1

    root = ET.fromstring(response.text)
    
    if len(root) == 0 or len(root[0]) == 0:
        return -1

    if type(root[0][0].find("molecule_type")) == type(None):
        return -1

    if root[0][0].find("molecule_type").text != "Small molecule":
        return -1

    if type(root[0][0].find("inorganic_flag")) == type(None) or type(root[0][0].find("inorganic_flag").text) == type(None):
        return -1
        
    return -1 * int(root[0][0].find("inorganic_flag").text)