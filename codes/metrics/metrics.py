from sklearn.metrics import precision_recall_fscore_support

# macro:宏平均  p = aver(p0+p1+p2+p3+...)   r = aver(r0+r1+r2+r3+...)  f = aver(f0+f1+f2+f3+...) 
# micro:微平均  
def er_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')

def re_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')

def gen_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')

def rc_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')

def p2so_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')

def gplinker_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')