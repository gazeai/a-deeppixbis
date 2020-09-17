import math
import numpy as np

import torch
from torch import nn

from threshold_dataloader import get_threshold_dataloaders
from deeppixdataloader import get_dataloaders
# from deeppix import SpoofBiFPN
from bifpn_3_coco import SpoofBiFPN_3 as SpoofBiFPN

from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd


gpus = [1]

def calculate_fpr(y_true, y_pred):
    true_spoof = 0  #### Spoof being 1
    false_real = 0  #### real being 0
    for i in range(len(y_true)):
        target = y_true[i]
        pred = y_pred[i]
        if target:
            true_spoof += 1
            if not pred:
                false_real += 1
    return false_real / true_spoof if true_spoof else 0

def calculate_fnr(y_true, y_pred):
    true_real = 0  #### Spoof being 1
    false_spoof = 0  #### real being 0
    for i in range(len(y_true)):   
        target = y_true[i]
        pred = y_pred[i]
        if not target:
            true_real += 1
            if pred:
                false_spoof += 1
    return false_spoof / true_real if true_real else 0

def calculate_eer(APCER, BPCER):
    return abs(APCER - BPCER)

def get_labels(probability, taw):
    pred = 0 if float(probability) < taw else 1
    return pred

def create_data(scores, taw):
    data = []
    for i, prob in enumerate(scores):
        pred = get_labels(prob, taw)
        data.append(pred)
        
    return data

def find_optimal_taw(scores, taw_range, taw_record, y_true):
    beta = 0.5
    for t in taw_range:
        y_hat = create_data(scores, t)
        APCER = calculate_fpr(y_true, y_hat)
        BPCER = calculate_fnr(y_true, y_hat)
        if t not in taw_record.keys():
            taw_record[t] = calculate_eer(APCER, BPCER)
    return min(taw_record, key=taw_record.get)


def get_th(test_data, network):
    y_true_th = []
    y_scores_th = []
    print('Generating scores from threshold set..')
    for idx, data in enumerate(tqdm(test_data)):
        im, lab = data
        with torch.no_grad():
            out = network.forward(im.cuda(gpus[0]))
        lab = [int(p) for p in lab]
        y_true_th.extend(lab)
        y_scores_th.extend(out.tolist())
        
    taw_list = np.arange(0.10, 1.00, .01).tolist()
    dict_taw = {}
    taw = find_optimal_taw(y_scores_th, taw_list, dict_taw, y_true_th)
    torch.cuda.empty_cache()
    return taw


def evaluate(model_label):
    network = SpoofBiFPN()
    if torch.cuda.device_count() > 1:
        network = nn.DataParallel(network, device_ids=gpus).cuda(gpus[0])
    cp = torch.load(f'./ckpts/{model_label}/model_1_93_2_11.pth')
    network.load_state_dict(cp['state_dict'])

    network.eval()
#('oulu', 'Protocol_1', 'val'), ('oulu', 'Protocol_2', 'val'), ('oulu', 'Protocol_3', 'val'), ('oulu', 'Protocol_4', 'val'),('msu', 'x', 'val'), ('mobile', 'x', 'val') 
    batch_size = 32
    datasets = [('msu', 'x', 'val'), ('gaze_front', 'x', 'val')]
    results = []

    for d in datasets:
        dataset_id = d[0]
        protocol = d[1]
        split = d[2]
        dataloader = get_dataloaders(data_type=dataset_id, hard_protocol=protocol, batch_size=batch_size)
        test = dataloader[split]

        dataloader_threshold = get_threshold_dataloaders(data_type=dataset_id, hard_protocol=protocol, batch_size=batch_size)
        test_threshold = dataloader_threshold[split]

        threshold = get_th(test_threshold, network) #0.12
        print(f'Evaluating {dataset_id}, Protocol: {protocol}, Threshold: {threshold}')

        y_true = []
        y_hat = []
        for idx, data in enumerate(tqdm(test)):
            im, lab = data
            with torch.no_grad():
                out = network.forward(im.cuda(gpus[0]))
            discrete_output = [1 if float(p) > threshold else 0 for p in out]
            lab = [int(p) for p in lab]
            y_true.extend(list(lab))
            y_hat.extend(discrete_output)

        acc = accuracy_score(y_true, y_hat) * 100
        apcer = calculate_fpr(y_true, y_hat) * 100
        bpcer = calculate_fnr(y_true, y_hat) * 100
        results.append({"dataset": dataset_id, "proto": protocol, "acc": acc, "apcer": apcer, "bpcer": bpcer, "th": threshold})
        df = pd.DataFrame(results)
        print(df)
#         df.to_csv(f'./results/{model_label}-no-align.csv', index=False)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    evaluate('bifpn-3-ss-msu-coco')