import h5py
import json
import numpy as np
import pandas as pd
import scipy.signal as spsig
from tqdm import tqdm
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

dir_path = "D:/Graduation_project/dataset/IEMOCAP_features_test/A/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


class ComParEExtractor(object):
    ''' 抽取comparE特征, 输入音频路径, 输出npy数组, 每帧130d
    '''

    def __init__(self, opensmile_tool_dir=None, downsample=10, tmp_dir='.tmp', no_tmp=False):
        ''' Extract ComparE feature
            tmp_dir: where to save opensmile csv file
            no_tmp: if true, delete tmp file
        '''
        # if not os.path.exists(tmp_dir):
            # os.makedirs(tmp_dir)
        if opensmile_tool_dir is None:
            opensmile_tool_dir = "D:/Graduation_project/opensmile-2.3.0"
        self.opensmile_tool_dir = opensmile_tool_dir
        self.tmp_dir = 'D:/Graduation_IEMOCAP/temp/'
        self.downsample = downsample
        self.no_tmp = no_tmp

    def __call__(self, wav):
        ''' Process the wav file to extract features.
            Parameters:
            wav (str): Path to the wav file.
            Returns:
            numpy array or None: Processed feature data or None if an error occurred.
        '''
        basename = os.path.basename(wav).split('.')[0]
        save_path = os.path.join(self.tmp_dir, basename + ".csv")
        # 设置命令提示符的代码页为 UTF-8 (65001)
        cmd_1 = 'chcp 65001'
        # 更改目录到指定的路径
        cmd_2 = 'cd /d D:/Graduation_project/opensmile-2.3.0/bin/Win32'
        # Update the command to be Windows-compatible:
        cmd_opensmile = 'SMILExtract_Release -C {}/config/ComParE_2016.conf ' \
              '-appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1 ' \
              '-I {} -lldcsvoutput {} -instname xx -O ? -noconsoleoutput 1'.format(
            self.opensmile_tool_dir,
            wav,
            save_path)

        cmd_combined = '{} && {} && {}'.format(cmd_1, cmd_2, cmd_opensmile)
        os.system(cmd_combined)

        try:
            df = pd.read_csv(save_path, delimiter=';')
            wav_data = df.iloc[:, 2:]
            if len(wav_data) > self.downsample:
                wav_data = spsig.resample_poly(wav_data, up=1, down=self.downsample, axis=0)
                if self.no_tmp:
                    os.remove(save_path)
            else:
                wav_data = None
                print(f'Error in {wav}, no feature extracted')
        except Exception as e:
            print(f'Error processing {wav}: {e}')
            wav_data = None

        return wav_data


def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label


def process_single_file(path, file_id):
    extractor = ComParEExtractor()
    wav_path = path
    file_name_with_extension = f"{file_id}.h5"

    # 确保文件路径存在
    if not os.path.exists(wav_path):
        print(f"文件 {wav_path} 不存在。")
        return
    # 提取特征
    feat = extractor(wav_path)
    # 存储特征
    feature_file_path = os.path.join('D:/Graduation_project/multimodal/mmin/MMIN-master/features',file_name_with_extension)
    with h5py.File(feature_file_path, 'w') as h5f:
        h5f[file_id] = feat
        print(f"特征已经被保存到 {feature_file_path}")

    with open('feature_path_info.json', 'w') as f:
        json.dump({'my_feature_path': feature_file_path}, f)


def make_all_comparE(config):
    extractor = ComParEExtractor()
    trn_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'tst')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    all_h5f = h5py.File(os.path.join(config['feature_root'], 'A', 'comparE.h5'), 'w')
    for utt_id in tqdm(all_utt_ids):
        ses_id = utt_id[4]
        if ses_id == '1':
            dialog_id = '_'.join(utt_id.split('_')[:-1])
            wav_path = os.path.join(config['data_root'], f'Session{ses_id}', 'sentences', 'wav', f'{dialog_id}',
                                    f'{utt_id}.wav')
            feat = extractor(wav_path)
            all_h5f[utt_id] = feat
    all_h5f.close()


def normlize_on_trn(config, input_file, output_file):
    h5f = h5py.File(output_file, 'w')
    in_data = h5py.File(input_file, 'r')
    for cv in range(1, 11):
        trn_int2name, _ = get_trn_val_tst(config['target_root'], cv, 'trn')
        trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
        all_feat = [in_data[utt_id][()] for utt_id in trn_int2name]
        all_feat = np.concatenate(all_feat, axis=0)
        mean_f = np.mean(all_feat, axis=0)
        std_f = np.std(all_feat, axis=0)
        std_f[std_f == 0.0] = 1.0
        cv_group = h5f.create_group(str(cv))
        cv_group['mean'] = mean_f
        cv_group['std'] = std_f
        print(cv)
        print("mean:", np.sum(mean_f))
        print("std:", np.sum(std_f))


if __name__ == '__main__':
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(
        "D:/Graduation_project/multimodal/mmin/MMIN-master/data/config/IEMOCAP_config.json")
    config = json.load(open(config_path))
    process_single_file('D:/Graduation_IEMOCAP/123.wav','123')
