import itertools
from torch.utils.data import Dataset
import os
import json

class TestMethodsDataset(Dataset):
    def __init__(self, rootdir, tokenizer) -> None:
        json_files = self.get_json_file_names(rootdir)
        records = [self.load_json_obj(os.path.join(rootdir,subdir)) for subdir in json_files]
        self.method_test_pairs = []
        super().__init__()

    def get_json_file_names(self, rootdir):
        result = []
        for _, subdirs, _ in os.walk(rootdir):
                for subdir in subdirs:
                        files = os.listdir(os.path.join(rootdir,subdir))
                        for file in files:
                                result.append(f'{subdir}/{file}')
        return result

    def load_json_obj(self,path):
            with open(path) as f:
                    data = json.load(f)
            return data

    def __len__(self):
        return len(self.method_test_pairs)
    
    def __getitem__(self, index):
        return super().__getitem__(index)




