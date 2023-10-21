

def f1_score_help(gold, pred):
    cnt_pred = 0
    cnt_right_pred = 0
    cnt_right_rec = 0
    cnt_gold = 0
    theta = 0.0001
    for single_gold, single_pred in zip(gold, pred):
        cnt_pred += len(single_pred)
        cnt_gold += len(single_gold)
        cnt_right_pred += len([element for element in single_gold if element in single_pred])
        cnt_right_rec += len([element for element in single_pred if element in single_gold])
    
    pre = cnt_right_pred/(cnt_pred + theta)
    rec = cnt_right_rec/(cnt_gold + theta)
    f1 = 2* pre * rec / (pre + rec + theta)

    return pre, rec, f1