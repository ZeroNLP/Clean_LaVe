import sys
import os

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)
import numpy as np
import torch
import torch.nn as nn

# from sentence_transformers import SentenceTransformer
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件夹


#Supporting Clustering with Contrastive Learning--sccl 无监督聚类 对比学习增强表示
def get_subj_obj_start(input_ids_arr,tokenizer,additional_index):
    """
    Function:
        to find the positon of the additional token, for example, 
        suppose we have a text: '<S:PERSON> Jobs </S:PERSON> is the founder of <O:ORGANIZATION> Apple </O:ORGANIZATION>'
        we gonna find the index of '<S:PERSON>' and '<O:ORGANIZATION>'
    Args:
    input_ids_arr like:
        tensor([[  101,  9499,  1071,  2149, 30522,  8696, 30522, 30534,  6874,  9033,
            4877,  3762, 30534, 10650,  1999, 12867,  1024,  5160,   102,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0],
            [  101,  2019, 21931, 17680,  2013, 11587, 30532,  2149, 30532, 14344,
            5016, 30537,  2406, 22517,  3361, 30537,  2006,  5958,  1010, 11211,
            2007, 10908,  2005,  1037,  2149,  3446,  3013,  2006,  9317,  1010,
            2992,  8069,  2008,  1996, 23902,  2013,  1996,  2149,  3847, 24185,
            2229,  2003, 24070,  1010, 16743,  2056,  1012,   102]])
    tokenizer:
        as named
    additional_index:
        the first index of the additional_special_tokens
    return:
         subj and obj start position
    """
    subj_starts = []
    obj_starts = []
    for input_ids in input_ids_arr:
        
        subj_start = -1
        obj_start = -1
        checked_id = []
        for idx,word_id in enumerate(input_ids):
            if subj_start!=-1 and obj_start!=-1:
                break
            if word_id>=additional_index:
                if word_id not in checked_id:
                    checked_id.append(word_id)
                    decoded_word = tokenizer.decode(word_id)
                    if decoded_word.startswith("<S:"):
                        subj_start = idx
                    elif decoded_word.startswith("<O:"):
                        obj_start = idx
        if subj_start==-1 or obj_start==-1:
            # if we did not find the speicial token, 
            # we set the start positoin to be 0
            if subj_start==-1:
                subj_start=0
            if obj_start==-1:
                obj_start=0
        subj_starts.append(subj_start)
        obj_starts.append(obj_start)
    return subj_starts,obj_starts

class SCCL_BERT(nn.Module):

    def __init__(self,bert_model,max_length,device,n_rel,open_bert = True,e_tags = []):
        """
        Args:
            bert_model: model loaded from sentence_transformer
            max_length: int, max encode length
            device: cuda device
            n_rel: i.e. Number_of_Relation, the number of classificaiton
            open_bert: bool, if open the bert (actually, we never fix the bert)
            e_tags: List[str],  special tokens that needed to add
        """
        super(SCCL_BERT, self).__init__()
        print("SCCL_BERT init")
        self.device = device
        self.max_length = max_length
        self.open_bert = open_bert
        
        
        self.tokenizer = bert_model[0].tokenizer
        self.sentbert = bert_model[0].auto_model
        self.additional_index = len(self.tokenizer) # original vocabulary size

        # add special tokens, such as <S:SUJ_TYPE> </S:SUJ_TYPE> 
        if len(e_tags)!=0:
            print("Add {num} special tokens".format(num=len(e_tags)))
            special_tokens_dict = {'additional_special_tokens': e_tags}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.sentbert.resize_token_embeddings(len(self.tokenizer))  # enlarge vocab
        
        self.embed_dim = self.sentbert.config.hidden_size # the embedding dimension of bert, e.g. 768
        
        if open_bert==False:
            for param in self.sentbert.parameters():
                param.requires_grad = False
            self.sentbert.eval()

        # the final full-connected layer, helping to classify
        self.out = nn.Linear(2*self.embed_dim,n_rel) 

    @staticmethod
    def cls_pooling(model_output):
        return model_output[0][:,0] # CLS token
    def get_embeddings(self, text_arr):
        """
        Args: 
            text_arr: List[str]
        return:
            the feature(CLS token) of the texts with shape (bs,d_model)
        """
        #这里的x都是文本
        feat_text= self.tokenizer.batch_encode_plus(text_arr, 
                                                    max_length=self.max_length+2,  # +2是因为CLS 和SEQ也算进去max_length的
                                                    return_tensors='pt', 
                                                    padding='longest',
                                                    truncation=True)
        for k,_ in feat_text.items():
            feat_text[k] = feat_text[k].to(self.device)
        self.sentbert.train()
        bert_output = self.sentbert.forward(**feat_text)

        #计算embedding (CLS)
        embedding = SCCL_BERT.cls_pooling(bert_output)

        return embedding
    def forward(self,texts):
        return self.out(self.get_embeddings_PURE(texts))
        
    def get_embeddings_PURE(self,text_arr):
        """
        From paper:
            A Frustratingly Easy Approach for Entity and Relation Extraction
            https://arxiv.org/abs/2010.12812
        Args: 
            text_arr: List[str]
        return:
            the feature(CLS token) of the texts with shape (bs,d_model*2) where d_model=768

        """
        feat_text= self.tokenizer.batch_encode_plus(text_arr, 
                                                    max_length=self.max_length+2, 
                                                    return_tensors='pt', 
                                                    padding='longest',
                                                    truncation=True)
        for k,_ in feat_text.items():
            feat_text[k] = feat_text[k].to(self.device)
        self.sentbert.train()

        # the index of the two entity mark
        ent1_spos,ent2_spos = get_subj_obj_start(feat_text['input_ids'],self.tokenizer,self.additional_index)

        bert_output = self.sentbert.forward(**feat_text)
        bert_output = bert_output[0]
        bs = bert_output.shape[0]
        assert len(ent1_spos)==len(ent2_spos)
        ent1_spos = torch.tensor(ent1_spos)
        ent2_spos = torch.tensor(ent2_spos)
        embedding1 = bert_output[[i for i in range(bs)],ent1_spos,:] # the feat represented by subj index
        embedding2 = bert_output[[i for i in range(bs)],ent2_spos,:] # the feat represented by sbj  index
        embeddings = torch.cat([embedding1,embedding2],dim = 1)
        return embeddings  # [bs, d_model * 2]




if __name__=="__main__":
    
    debug_stop = 1

    