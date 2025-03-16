import numpy as np
# 读取npy文件
np.set_printoptions(threshold=np.inf)
data = np.load('/home/hetianqu/multimodal/mmin/MMIN-master/checkpoints/mmin_IEMOCAP_block_5_run1/1/test_label.npy')
# 查看文件内容
print(data)