import os
import json
import numpy as np
from opts.get_opts import Options
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from models.utils.config import OptConfig
import sys

args = sys.argv[:]
import json
import re


def eval_miss(model, val_iter):
    model.eval()
    total_pred = []
    total_label = []
    total_miss_type = []
    for _, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        total_pred.append(pred)
        total_label.append(label)

    # calculate metrics
    return pred


if __name__ == '__main__':
    test_other = True
    in_men = True
    flag = False
    total_cv = 10
    gpu_ids = [0]
    comparison_value = sys.argv[1]
    model_choose = sys.argv[2]
    miss_type = sys.argv[3]
    anger = 0
    happy = 0
    sad = 0
    netural = 0
    print(comparison_value)
    sub_path = "checkpoints/"
    ckpt_path = os.path.join(sub_path, model_choose)

    my_file_path = sys.argv[4]

    with open('path_info.json', 'w') as f:
        json.dump({'my_file_path': my_file_path}, f)

    with open('name_info.json', 'w') as f:
        json.dump({'my_file_name': comparison_value}, f)

    config = json.load(open(os.path.join(ckpt_path, 'train_opt.conf')))
    file_types = ['trn_int2name.npy', 'val_int2name.npy', 'tst_int2name.npy']
    opt = OptConfig()
    opt.load(config)
    matched_info = []

    if test_other:
        opt.dataset_mode = 'dummy'

    opt.gpu_ids = gpu_ids
    setattr(opt, 'in_mem', in_men)
    print('初始化模型中...')
    model = create_model(opt)
    model.setup(opt)
    results = []

    for cv in range(1, 1 + total_cv):
        opt.cvNo = cv
        tst_dataloader = create_dataset_with_args(opt, set_name='tst')
        model.load_networks_cv(os.path.join(ckpt_path, str(cv)))
        model.eval()
        if test_other:
            res_other = eval_miss(model, tst_dataloader)

        numbers = res_other[0]
        print(numbers)
        if numbers == 0:
            anger = anger + 1
        if numbers == 1:
            happy = happy + 1
        if numbers == 2:
            netural = netural + 1
        if numbers == 3:
            sad = sad + 1

    anger = anger + 2

    variables = {'anger': anger, 'happy': happy, 'sad': sad, 'n': netural}
    max_var_name = max(variables, key=variables.get)

    print('注：语音、图像和文本三个模态分别以字母a、v、l表示，缺失的模态用字母z表示\n')
    print('情感识别结果共有四种标签，分别为愤怒、快乐、悲伤和中立\n')

    if test_other:
        if max_var_name == 'anger':
            print(f'您上传的片段为{comparison_value},模态缺失情况为{miss_type},情感识别结果为<愤怒>')
        elif max_var_name == 'happy':
            print(f'您上传的片段为{comparison_value},模态缺失情况为{miss_type},情感识别结果为<快乐>')
        elif max_var_name == 'n':
            print(f'您上传的片段为{comparison_value},模态缺失情况为{miss_type},情感识别结果为<中立>')
        elif max_var_name == 'sad':
            print(f'您上传的片段为{comparison_value},模态缺失情况为{miss_type},情感识别结果为<悲伤>')