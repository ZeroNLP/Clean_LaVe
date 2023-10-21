import json
import pickle  

e_tag = ["<e1>", "</e1>", "<e2>", "</e2>"]

with open("/home/jwwang/URE_multi/NL/config/e_tag_path.pkl", 'wb') as file:
    pickle.dump(e_tag, file)