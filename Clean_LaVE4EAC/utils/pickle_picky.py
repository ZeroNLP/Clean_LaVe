import pickle

def pickle_save(obj,path_name):
    print("保存到:",path_name)
    with open(path_name,'wb') as file:
        pickle.dump(obj,file)

def pickle_load(path_name: object) -> object:
    with open(path_name,'rb') as file:
        return pickle.load(file)