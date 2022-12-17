from _utils import get_json_file_names

def convert_jsons_to_jsonl(jsonFiles, output_name):
    lines = []
    with open(output_name, "a+") as output_file:
        for idx, file in enumerate(jsonFiles):
            content = read_file_content(file)
            output_file.write(content)
            output_file.write("\n")
        

def read_file_content(file):
    with open(file, "r") as f:
        data = f.read()
    return data

files = get_json_file_names(rootdir="E:\Fisierele mele\Facultate\DNN\Project\Code\methods2test\corpus\json\\train")
output_name = "E:\Fisierele mele\Facultate\DNN\Project\Code\methods2test\corpus\json\\train.jsonl"
#files = files[0:100000]
convert_jsons_to_jsonl(files, output_name)
print("Finished processing all files")