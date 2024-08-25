import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import global_parameters_set
from torch.utils import data
from data_load_utils import DTIDataset, drug_graph_and_protein_sequence_collate_fn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score, auc, precision_recall_curve

from models import CrossSequenceModel


def load_data(config):
    dataframe_train = pd.read_csv("./datasets/" + config['dataset_name'] + "/train.csv")
    dataframe_val = pd.read_csv("./datasets/" + config['dataset_name'] + "/val.csv")
    dataframe_test = pd.read_csv("./datasets/" + config['dataset_name'] + "/test.csv")
    dataset_all_file_path = "./datasets/" + config['dataset_name'] + "/all.csv"

    train_data_set = DTIDataset(dataframe_train)
    val_data_set = DTIDataset(dataframe_val)
    test_data_set = DTIDataset(dataframe_test)

    data_loader_params_train = {'batch_size': config['batch_size'],
                          'shuffle': True,
                          'num_workers': config['workers'],
                          'drop_last': False,
                          'collate_fn': drug_graph_and_protein_sequence_collate_fn}
    data_loader_params_val_and_test = {'batch_size': config['batch_size'],
                                'shuffle': False,
                                'num_workers': config['workers'],
                                'drop_last': False,
                                'collate_fn': drug_graph_and_protein_sequence_collate_fn}
    train_data_loader = data.DataLoader(train_data_set, **data_loader_params_train)
    val_data_loader = data.DataLoader(val_data_set, **data_loader_params_val_and_test)
    test_data_loader = data.DataLoader(test_data_set, **data_loader_params_val_and_test)

    return train_data_loader, val_data_loader, test_data_loader


def test(data_generator, model, dataset_name, epoch_num, is_test=False):
    model_name = type(model).__name__
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for i, (atom_num, drug_graph, protein_sequence_length, protein_one_hot_index_list, label) in enumerate(data_generator):
        logits = torch.squeeze(model(atom_num.cuda(), drug_graph.cuda(), protein_sequence_length.cuda(), protein_one_hot_index_list.long().cuda()))
        loss_fct = nn.BCEWithLogitsLoss()

        label = label.float().cuda()

        loss = loss_fct(logits, label).item()

        loss_accumulate += loss
        count += 1

        logits = F.sigmoid(logits).detach().cpu().numpy()
        label_ids = label.cpu().numpy()
        y_pred = y_pred + logits.tolist()
        y_label = y_label + label_ids.tolist()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision, recall, thresholds = precision_recall_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    # thred_optim = thresholds[5:][np.argmax(f1[5:])] if len(f1) >= 5 else thresholds[np.argmax(f1)]
    thred_optim = np.float64(0.5)

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return accuracy1, roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), loss


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    torch.backends.cudnn.benchmark = True

    dti_config = global_parameters_set()
    train_data_loader, val_data_loader, test_data_loader = load_data(dti_config)

    model = CrossSequenceModel(**dti_config)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=dti_config['learning_rate'], weight_decay=1e-2)
    loss_function = nn.BCEWithLogitsLoss()

    val_weights_file = "./weights/" + dti_config['dataset_name'] + '_' + type(model).__name__ + "_val_weights_sigmoid.pth"
    test_weights_file = "./weights/" + dti_config['dataset_name'] + '_' + type(model).__name__ + "_test_weights_sigmoid.pth"

    if os.path.exists(val_weights_file):
        checkpoint = torch.load(val_weights_file)
        best_val_auroc = checkpoint['val_auroc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        print("loss: " + str(checkpoint['val_loss']))
        print("AUROC:" + str(checkpoint['val_auroc']))
        print("AUPRC:" + str(checkpoint['val_auprc']))
    else:
        best_val_auroc = 0

    if os.path.exists(test_weights_file):
        checkpoint = torch.load(test_weights_file)
        best_test_auroc = checkpoint['test_auroc']
    else:
        best_test_auroc = 0

    last_best_val_auroc = best_val_auroc
    last_best_test_auroc = best_test_auroc

    best_model = copy.deepcopy(model)

    for epoch in range(dti_config['train_epoch']):
        model.train()
        for i, (atom_num, drug_graph, protein_sequence_length, protein_one_hot_index_list, label) in enumerate(train_data_loader):
            score = model(atom_num.cuda(), drug_graph.cuda(), protein_sequence_length.cuda(), protein_one_hot_index_list.long().cuda())
            predict = torch.squeeze(score)
            label = label.float().cuda()

            loss = loss_function(predict, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 0 or i % 10 == 0:
                print('Training at Epoch ' + str(epoch + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))

        found_better_val_model = False
        with torch.set_grad_enabled(False):
            val_acc, val_auc, val_auprc, val_f1, val_loss = test(val_data_loader, model, dti_config['dataset_name'], epoch + 1, False)
            if val_auc > best_val_auroc:
                found_better_val_model = True
                best_val_auroc = val_auc
                best_model = copy.deepcopy(model)
                state = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_auroc": val_auc,
                    "val_auprc": val_auprc,
                    "val_f1": val_f1
                }
                torch.save(state, val_weights_file)
            print('Validation at Epoch ' + str(epoch + 1) + ' , AUROC: ' + str(val_auc) + ' , AUPRC: ' + str(
                val_auprc) + ' , ACC: ' + str(val_acc) + ' , F1: ' + str(val_f1) + ' , loss: ' + str(val_loss))

        with torch.set_grad_enabled(False):
            test_acc, test_auc, test_auprc, test_f1, test_loss = test(test_data_loader, best_model, dti_config['dataset_name'], epoch + 1, True)
            if test_auc > best_test_auroc:
                best_test_auroc = test_auc
                state = {
                    "state_dict": best_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_auroc": test_auc,
                    "test_auprc": test_auprc,
                    "test_f1": test_f1
                }
                torch.save(state, test_weights_file)
            print('Test at Epoch ' + str(epoch + 1) + ' , AUROC: ' + str(test_auc) + ' , AUPRC: ' + str(
                test_auprc) + ' , ACC: ' + str(test_acc) + ' , F1: ' + str(test_f1) + ' , loss: ' + str(test_loss))
