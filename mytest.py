import os
import itertools
import json

def get_json_file_names(rootdir):
        result = []
        for path, subdirs, files in os.walk(rootdir):
                for subdir in subdirs:
                        files = os.listdir(os.path.join(rootdir,subdir))
                        for file in files:
                                result.append(f'{subdir}/{file}')
        return result

def load_json_obj( path):
        with open(path) as f:
                data = json.load(f)
        return data


rootdir = "E:\Fisierele mele\Facultate\DNN\Project\Code\methods2test\corpus\json\eval"
json_files = get_json_file_names(rootdir)
records = [load_json_obj(os.path.join(rootdir,subdir)) for subdir in json_files]
test = records[0]["src_fm"]
print(test)