import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

from data import create_dataset, create_dataset_with_args
from models import create_model
from models.utils.config import OptConfig
from opts.get_opts import Options
from utils.logger import get_logger, ResultRecorder
args = sys.argv[:]


def eval_miss_input(model, val_iter, name, miss):
    model.eval()
    total_pred = []
    total_label = []
    total_miss_type = []
    res = None
    for _, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        int2name = data['int2name']
        miss_type = np.array(data['miss_type'])
        total_pred.append(pred)
        total_label.append(label)
        total_miss_type.append(miss_type)

        try:
            name_index = int2name.index(name)  # 直接查找名字对应的索引
        except ValueError:
            continue  # 如果名字不在列表中，继续下一个循环

        # 检查 miss 类型是否存在于指定范围内
        for k in range(name_index, name_index + 5):
            if k < len(miss_type) and miss_type[k] == miss:
                res = pred[k]
                lab = label[k]
                return res, lab  # 找到结果后直接返回

    return res


def output_eval_all(model, val_iter):
    model.eval()
    total_pred = []
    total_label = []
    for _, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        total_pred.append(pred)
        total_label.append(label)

    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1 = f1_score(total_label, total_pred, average='macro')
    cm = confusion_matrix(total_label, total_pred)

    return acc, uar, f1, cm, total_pred


if __name__ == '__main__':
    test_miss = False
    test_base = False
    in_men = True
    flag = False
    total_cv = 10
    gpu_ids = [0]
    matched_indice = []
    comparison_value = sys.argv[1]
    model_choose = sys.argv[2]
    lab_miss = None
    l_miss = None

    miss_type = sys.argv[3]
    if miss_type != 'avl':
        test_miss = True
        test_base = False
    else:
        test_miss = False
        test_base = True

    print(miss_type)
    print(comparison_value)
    sub_path = "checkpoints/"
    ckpt_path = os.path.join(sub_path, model_choose)
    data_path = "D:\\Graduation_project\\dataset\\IEMOCAP\\IEMOCAP_features_2021\\target"
    config = json.load(open(os.path.join(ckpt_path, 'train_opt.conf')))
    opt = OptConfig()
    opt.load(config)
    matched_info = []
    file_types = ['trn_int2name.npy', 'val_int2name.npy', 'tst_int2name.npy']
    if test_base:
        opt.dataset_mode = 'multimodal'
    if test_miss:
        opt.dataset_mode = 'multimodal_miss'

    print('检测输入数据中:')
    for i in range(1,11):
        sub_dir_path = os.path.join(data_path, str(i))
        for file_type in file_types:
            file_path = os.path.join(sub_dir_path, file_type)
            if os.path.exists(file_path):
                # 加载.npy文件
                npy_data = np.load(file_path)
                # 处理数据
                for idx, item in enumerate(npy_data):
                    if item[0].decode('utf-8') == comparison_value:
                        print(f"Found {comparison_value} in {file_path}, at index {idx}")
                        matched_info.append((file_type.split('_')[0], i, idx))
                        break
                if matched_info:
                    break
        if matched_info:
            break
    match = matched_info[0]
    cv = match[1]
    idx = match[2]
    dataset =match[0]
    opt.gpu_ids = gpu_ids
    setattr(opt, 'in_mem', in_men)
    print('初始化模型中...')
    model = create_model(opt)
    model.setup(opt)
    results = []

    opt.cvNo = cv
    tst_dataloader = create_dataset_with_args(opt, set_name=dataset)
    model.load_networks_cv(os.path.join(ckpt_path, str(cv)))
    model.eval()
    if test_base:
        acc, uar, f1, cm, res = output_eval_all(model, tst_dataloader)
    if test_miss:
        res_miss, lab_miss = eval_miss_input(model, tst_dataloader, comparison_value, miss_type)

    print('\n\n\n')

    print('注：语音、图像和文本三个模态分别以字母a、v、l表示，缺失的模态用字母z表示\n')
    print('情感识别结果共有四种标签，分别为愤怒、快乐、悲伤和中立\n')

    if lab_miss == 0:
        l_miss = '愤怒'
    elif lab_miss == 1:
        l_miss = '快乐'
    elif lab_miss == 2:
        l_miss = '中立'
    elif lab_miss == 3:
        l_miss = '悲伤'

    if test_base:
        if res[idx] == 0:
            print(f'您上传的片段为{comparison_value},模态缺失情况为{miss_type},情感识别结果为<愤怒>')
        elif res[idx] == 1:
            print(f'您上传的片段为{comparison_value},模态缺失情况为{miss_type},情感识别结果为<快乐>')
        elif res[idx] == 2:
            print(f'您上传的片段为{comparison_value},模态缺失情况为{miss_type},情感识别结果为<中立>')
        elif res[idx] == 3:
            print(f'您上传的片段为{comparison_value},模态缺失情况为{miss_type},情感识别结果为<悲伤>')

    if test_miss:
        if res_miss == 0:
            print(f'您上传的片段为{comparison_value},模态缺失情况为{miss_type},情感识别结果为<愤怒>,原标签为{l_miss}')
        elif res_miss == 1:
            print(f'您上传的片段为{comparison_value},模态缺失情况为{miss_type},情感识别结果为<快乐>,原标签为{l_miss}')
        elif res_miss == 2:
            print(f'您上传的片段为{comparison_value},模态缺失情况为{miss_type},情感识别结果为<中立>,原标签为{l_miss}')
        elif res_miss == 3:
            print(f'您上传的片段为{comparison_value},模态缺失情况为{miss_type},情感识别结果为<悲伤>,原标签为{l_miss}')
