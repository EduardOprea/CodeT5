import argparse
import json
import os
from _utils import get_json_file_names
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='E:\Fisierele mele\Facultate\DNN\Project\Code\methods2test\\corpus\\json\\eval')
    parser.add_argument("--metadata_dir", type=str, default='E:\Fisierele mele\Facultate\DNN\Project\Code\methods2test\dataset\\eval')
    parser.add_argument("--output_dir", type=str, default='E:\Fisierele mele\Facultate\DNN\Project\Code\methods2test\corpus\\rename_fm_and_args\\eval')
    args = parser.parse_args()
    return args
def get_json_obj(file):
    with open(file,"r") as f:
        data = json.load(f)
    return data
def get_repo_and_testcase_ids(path):
    splits = path.split("\\")
    filename = splits[len(splits)-1]
    name_splits = filename.split("_")
    if len(name_splits) != 3 or name_splits[2] != 'corpus.json':
        raise ValueError("Parsing the json file name went wrong, name -> ", filename)
    
    return name_splits[0], name_splits[1]
def get_metadata_obj(metadata_dir, repo_id, testcase_id):
    metadata_obj = get_json_obj(os.path.join(metadata_dir, repo_id, repo_id + "_" + testcase_id + ".json"))
    return metadata_obj

def check_exists(metadata_dir, repo_id, testcase_id):
    metadata_filepath = os.path.join(metadata_dir, repo_id, repo_id + "_" + testcase_id + ".json")
    if os.path.exists(metadata_filepath) == False:
        raise ValueError("Metadata file does not exists ", metadata_filepath)

def get_parameters_list(signature):
    try:
        params_names = []
        signature = signature.replace("\n","")
        signature = signature.replace("(","")
        signature = signature.replace(")","")
        if len(signature) == 0:
            return params_names
        
        signature_params = signature.split(",")

        for param in signature_params:
            param = param.strip()
            type_and_name = param.split(" ")
            params_names.append(type_and_name[len(type_and_name)-1])

        return params_names
    except:
        print("Exception ocurred for signature : ", signature)
        raise ValueError("Eroare")
    


def simplify_sample(args, file):
    sample_obj = get_json_obj(file)
    repo_id, testcase_id = get_repo_and_testcase_ids(file)
    metadata_obj = get_metadata_obj(args.metadata_dir, repo_id, testcase_id)

    focal_method_name = metadata_obj["focal_method"]["identifier"]
    params_focal_method = get_parameters_list(metadata_obj["focal_method"]["parameters"])
     
    src_fm = sample_obj["src_fm"]    
    target_test = sample_obj["target"]

    src_fm = src_fm.replace(focal_method_name, "focalMethod")
    target_test = target_test.replace(f'{focal_method_name}(', "focalMethod(")
    

    for idx, param in enumerate(params_focal_method):
        src_fm = src_fm.replace(param,f"arg{idx}")

    
    sample_obj["target"] = target_test    
    sample_obj["src_fm"] = src_fm

    write_json(os.path.join(args.output_dir, repo_id),f'{repo_id}_{testcase_id}_corpus.json', sample_obj)



def write_json(dir, filename, data):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir, filename), "w") as f:
        json.dump(data, f)
if __name__ == "__main__":
    args = parse_args()
    files = get_json_file_names(args.dataset_dir)
    files = files[0:100000]
    for file in tqdm(files):
        simplify_sample(args, file)

    
