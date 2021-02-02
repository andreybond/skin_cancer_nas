# https://towardsdatascience.com/metrics-for-imbalanced-classification-41c71549bbb5

def show_metrics(y_true, y_score):
    # True positive
    tp = np.sum(y_true * y_score)
    # False positive
    fp = np.sum((y_true == 0) * y_score)
    # True negative
    tn = np.sum((y_true==0) * (y_score==0))
    # False negative
    fn = np.sum(y_true * (y_score==0))

    # True positive rate (sensitivity or recall)
    tpr = tp / (tp + fn)
    # False positive rate (fall-out)
    fpr = fp / (fp + tn)
    # Precision
    precision = tp / (tp + fp)
    # True negatvie tate (specificity)
    tnr = 1 - fpr
    # F1 score
    f1 = 2*tp / (2*tp + fp + fn)
    # ROC-AUC for binary classification
    auc = (tpr+tnr) / 2
    # MCC
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    print("True positive: ", tp)
    print("False positive: ", fp)
    print("True negative: ", tn)
    print("False negative: ", fn)

    print("True positive rate (recall): ", tpr)
    print("False positive rate: ", fpr)
    print("Precision: ", precision)
    print("True negative rate: ", tnr)
    print("F1: ", f1)
    print("ROC-AUC: ", auc)
    print("MCC: ", mcc)


import torch 
import torch.nn as nn

class LogReg(nn.Module):
    def __init__(self, in_dim):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y_pred = self.sigmoid(self.linear(x))
        return y_pred

#show_metrics(y_test, (y_pred > 0.5).astype(np.int))

def roc_auc_score(y_true, y_pred):
    unique_vals = list(np.sort(np.unique(y_pred))) + [1]
    area_under_curve = 0
    fpr_points, tpr_points = [], []
    for i, th in enumerate(unique_vals):
        y_pred_th = np.zeros(len(y_pred))
        y_pred_th[y_pred >= th] = 1.0
        
        tp = np.sum(y_test * y_pred_th)
        fp = np.sum((y_test == 0) * y_pred_th)
        tn = np.sum((y_test==0) * (y_pred_th==0))
        fn = np.sum(y_test * (y_pred_th==0))
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        fpr_points.append(fpr)
        tpr_points.append(tpr)
        
        if i > 0:
            area_under_curve += (fpr_prev - fpr) * tpr
        fpr_prev = fpr
        
    plt.figure(figsize=(5, 5), dpi=100)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    sns.lineplot(fpr_points, tpr_points)
    plt.show()
        
    return area_under_curve

print(roc_auc_score(y_test, y_pred))