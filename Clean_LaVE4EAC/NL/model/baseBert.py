import sys
import os

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)
import numpy as np
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件夹


class EAE_BERT(nn.Module):

    def __init__(self,bert_model, max_length, device, n_rel, open_bert = True):
        """
        Args:
            bert_model: model loaded from sentence_transformer
            max_length: int, max encode length
            device: cuda device
            n_rel: i.e. Number_of_Relation, the number of classificaiton
            open_bert: bool, if open the bert (actually, we never fix the bert)
        """
        super(EAE_BERT, self).__init__()
        print("SCCL_BERT init", flush=True)
        self.device = device
        self.max_length = max_length
        self.open_bert = open_bert
        
        
        self.tokenizer = bert_model[0].tokenizer
        self.sentbert = bert_model[0].auto_model
        self.additional_index = len(self.tokenizer) # original vocabulary size
        
        self.embed_dim = self.sentbert.config.hidden_size # the embedding dimension of bert, e.g. 768
        self.n_rel = n_rel

        if open_bert==False:
            for param in self.sentbert.parameters():
                param.requires_grad = False

        # the final full-connected layer, helping to classify
        self.out = nn.Linear(2* self.embed_dim, n_rel) # 从最低维 开始匹配 找到满足线性层输入的n维度

    
    def combine_loss(self, loss_neg):
        loss = loss_neg
        return loss
    
    def get_trigger_arg_be(self, text_pieces, trigger_be, arg_be):
        # trigger
        trigger_b_text = []
        trigger_start = []
        trigger_e_text = []
        trigger_end = []
        for piece, t_be in zip(text_pieces, trigger_be):
            trigger_b_text.append(" ".join(piece[0:t_be[0]]))
            trigger_e_text.append(" ".join(piece[0:t_be[1] + 1]))
            trigger_start.append(len(self.tokenizer.tokenize(trigger_b_text[-1])))
            trigger_end.append(len(self.tokenizer.tokenize(trigger_e_text[-1])))


            # print(" ".join(piece[0:t_be[0]]))
            # print(" ".join(piece[0:t_be[1] + 1]))
            # print(" ".join(piece[t_be[0]: t_be[1] + 1]).strip())
            # print("recovered trigger: {}".format(self.tokenizer.convert_tokens_to_string(
            #                                      self.tokenizer.tokenize(" ".join(piece))[trigger_start[-1]: trigger_end[-1]])
            # )
            # )
            # tokenizer的tokenize会增加一句话的空格
            # assert(" ".join(piece[t_be[0]: t_be[1] + 1]).strip() == 
            #        self.tokenizer.convert_tokens_to_string(
            #                                      self.tokenizer.tokenize(" ".join(piece))[trigger_start[-1]: trigger_end[-1]]).strip()
            #                                      )
            # print(" ".join(piece[0:t_be[0]]))
            # print(" ".join(piece[0:t_be[1] + 1]))
            # print("recovered trigger: {}".format(self.tokenizer.convert_tokens_to_string(
            #                                      self.tokenizer.tokenize(" ".join(piece))[trigger_start[-1]: trigger_end[-1]])
            # )
            # )
        
        #! begin_end有区别
        # trigger_b_token = self.tokenizer.batch_encode_plus(
        #     trigger_b_text,
        #     max_length=self.max_length+2,
        #     return_tensors='pt', 
        #     padding='longest',
        #     truncation=True
        # )
        # trigger_start = [len(ids) for ids in trigger_b_token['input_ids']]

        # trigger_e_token = self.tokenizer.batch_encode_plus(
        #     trigger_e_text,
        #     max_length=self.max_length+2,
        #     return_tensors='pt', 
        #     padding='longest',
        #     truncation=True
        # )
        # trigger_end = [len(ids) for ids in trigger_e_token['input_ids']]

        # argument:
        arg_b_text = []
        arg_e_text = []
        arg_start = []
        arg_end = []
        for piece, a_be in zip(text_pieces, arg_be):
            arg_b_text.append(" ".join(piece[0:a_be[0]]))
            arg_e_text.append(" ".join(piece[0:a_be[1] + 1]))
            arg_start.append(len(self.tokenizer.tokenize(arg_b_text[-1])))
            arg_end.append(len(self.tokenizer.tokenize(arg_e_text[-1])))

            # print(" ".join(piece[0:a_be[0]]))
            # print(" ".join(piece[0:a_be[1] + 1]))
            # print(" ".join(piece[a_be[0]: a_be[1] + 1]).strip())
            # print("recovered argument: {}".format(self.tokenizer.convert_tokens_to_string(
            #                                      self.tokenizer.tokenize(" ".join(piece))[arg_start[-1]: arg_end[-1]])
            # )
            # )




        # arg_b_token = self.tokenizer.batch_encode_plus(
        #     arg_b_text,
        #     max_length=self.max_length+2,
        #     return_tensors='pt', 
        #     padding='longest',
        #     truncation=True
        # )
        # arg_start = [len(ids) for ids in arg_b_token['input_ids']]

        # arg_e_token = self.tokenizer.batch_encode_plus(
        #     arg_e_text,
        #     max_length=self.max_length+2,
        #     return_tensors='pt', 
        #     padding='longest',
        #     truncation=True
        # )
        # arg_end = [len(ids) for ids in arg_e_token['input_ids']]



        return trigger_start, trigger_end, arg_start, arg_end



    def get_embeddings(self, text_arr, text_pieces, trigger_be, arg_be):
        """
        Args: 
            text_arr: List[str]
        return:
            the feature(CLS token) of the texts with shape (bs,d_model)
        """
        feat_text= self.tokenizer.batch_encode_plus(text_arr, 
                                                    max_length=self.max_length+2,  # +2是因为CLS 和SEP也算进去max_length的
                                                    return_tensors='pt', 
                                                    padding='longest',
                                                    truncation=True)

        for k,_ in feat_text.items():
            feat_text[k] = feat_text[k].to(self.device)
        bert_output = self.sentbert.forward(**feat_text)[0]

        trigger_start, trigger_end, arg_start, arg_end = self.get_trigger_arg_be(
            text_pieces= text_pieces, 
            trigger_be = trigger_be, 
            arg_be = arg_be
        )

        # 离散， 条件语句， !!!!copy/新建 tensor（不共享grad, 两个tensor 独立）
        # 离散下标 可导, 但是argmax不可导，取最大的创建one-hot向量，新向量和之前没直接关系；取下标部分的原向量，可导，并没有创建新的向量。
        batch_t = []
        batch_e = []
        cls = []
        bs = bert_output.shape[0]
        # embedding1 = bert_output[[i for i in range(bs)],trigger_start,:] # the feat represented by subj index
        # embedding2 = bert_output[[i for i in range(bs)],arg_start,:] # the feat represented by sbj  index
        # return torch.cat([embedding1,embedding2],dim = 1)

        for index, t_start, t_end, a_start, a_end in zip(
            np.arange(bs), trigger_start, trigger_end, arg_start, arg_end
        ):
            batch_t.append(torch.mean(bert_output[index ,t_start:t_end, :], 0))
            batch_e.append(torch.mean(bert_output[index ,a_start:a_end, :], 0))

        return torch.cat([torch.stack(batch_t), torch.stack(batch_e)], dim=1)
        


    