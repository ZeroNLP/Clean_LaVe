import  pickle

def save(obj,path_name):
    print("保存到:",path_name)
    with open(path_name,'wb') as file:
        pickle.dump(obj,file)

def load(path_name: object) -> object:
    
    with open(path_name,'rb') as file:
        return pickle.load(file)