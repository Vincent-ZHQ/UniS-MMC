
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, roc_auc_score


def collect_metrics(dataset, y_true, y_pred):
    if dataset == 'Food101':
        acc = accuracy_score(y_true, y_pred.argmax(1))
        tp2acc = top_k_accuracy_score(y_true, y_pred, k=2)
        tp5acc = top_k_accuracy_score(y_true, y_pred, k=5)
        wf1 = f1_score(y_true, y_pred.argmax(1), average='weighted')
        uf1 = f1_score(y_true, y_pred.argmax(1), average='macro')

        eval_results = {
            "acc": round(acc, 4),
            "tp2acc": round(tp2acc, 4),
            "tp5acc": round(tp5acc, 4),
            "wf1": round(wf1, 4),
            "uf1": round(uf1, 4)
        }

    elif dataset == 'VSNLI':
        acc = accuracy_score(y_true, y_pred.argmax(1))
        wf1 = f1_score(y_true, y_pred.argmax(1), average='weighted')
        uf1 = f1_score(y_true, y_pred.argmax(1), average='macro')

        eval_results = {
            "acc": round(acc, 4),
            "wf1": round(wf1, 4),
            "uf1": round(uf1, 4)
        }

    elif dataset == 'N24News':
        acc = accuracy_score(y_true, y_pred.argmax(1))
        wf1 = f1_score(y_true, y_pred.argmax(1), average='weighted')
        uf1 = f1_score(y_true, y_pred.argmax(1), average='macro')

        eval_results = {
            "acc": round(acc, 4),
            "wf1": round(wf1, 4),
            "uf1": round(uf1, 4)
        }

    elif dataset == 'HatefulMemes':
        acc = accuracy_score(y_true, y_pred.argmax(1))
        wf1 = f1_score(y_true, y_pred.argmax(1), average='weighted')
        uf1 = f1_score(y_true, y_pred.argmax(1), average='macro')
        auc = roc_auc_score(y_true, y_pred[:, 1])

        eval_results = {
            "acc": round(acc, 4),
            "wf1": round(wf1, 4),
            "uf1": round(uf1, 4),
            "auc": round(auc, 4)
        }

    elif dataset == 'BRCA':
        acc = accuracy_score(y_true, y_pred.argmax(1))
        wf1 = f1_score(y_true, y_pred.argmax(1), average='weighted')
        uf1 = f1_score(y_true, y_pred.argmax(1), average='macro')

        eval_results = {
            "acc": round(acc, 4),
            "wf1": round(wf1, 4),
            "uf1": round(uf1, 4)
        }

    else:
        acc = accuracy_score(y_true, y_pred.argmax(1))
        f1 = f1_score(y_true, y_pred.argmax(1))
        auc = roc_auc_score(y_true, y_pred[:, 1])

        eval_results = {
            "acc": round(acc, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4)
        }

    return eval_results

