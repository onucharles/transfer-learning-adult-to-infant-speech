from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def plot_roc_curve(true_labels, pred_probs, pos_label=3):
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs, pos_label=pos_label)

    # TODO can I do this in a more resilient way?
    # subtract 1 since this fxn doesn't let one specify positive label.
    roc_auc = np.round(roc_auc_score(true_labels, 1 - pred_probs), 2)
    print('Area under the ROC curve = {0}'.format(roc_auc))

    plt.figure()
    plt.plot(fpr, tpr, markersize=4, label= '(AUC={0})'.format(roc_auc), linewidth=2, color='g')
    plt.plot([0,1],[0,1], linewidth=1.5, color='b', linestyle='--', label='Random')
    plt.xlabel('1 - Specificity', fontsize=13)
    plt.ylabel('Sensitivity', fontsize=13)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.grid()
    plt.legend(fontsize=13)