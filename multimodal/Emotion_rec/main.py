import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from emo_opts import parse_opts

from core.model import generate_model
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform
from core.dataset import get_validation_set, get_data_loader

from transforms.temporal import TSN
from transforms.target import ClassLabel

from torch.cuda import device_count
import torch
from emo_test import test
from emo_tools import processing
from tensorboardX import SummaryWriter
import argparse

os.environ['CUDA_VISIBLE_DIVICES'] = '0'

result1 = None


def main(name):
    image_path = "data/Joy/" + name
    audio_path = "data/Joy/" + name + "/mp3/mp3"
    test_flag = True
    log_dir = "save_30.pth"

    opt = parse_opts()
    opt.device_ids = list(range(device_count()))
    local2global_path(opt)
    model, parameters = generate_model(opt)

    criterion = get_loss(opt)
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)

    writer = SummaryWriter(logdir=opt.log_path)

    # test
    if test_flag:
        spatial_transform = get_spatial_transform(opt, 'test')
        temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
        target_transform = ClassLabel()
        validation_data = get_validation_set(image_path, audio_path, opt, spatial_transform, temporal_transform,
                                             target_transform)
        val_loader = get_data_loader(opt, validation_data, shuffle=False)

        checkpoint = torch.load(log_dir, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        result, result2 = test(1, val_loader, model, criterion, opt, writer, optimizer)
        return result, result2

    writer.close()


def output_result(path):
    name = processing.video(path)
    return name


def results(name):
    result1, result2 = main(name)
    return result1, result2


def parse_args():
    parser = argparse.ArgumentParser(description="Video processing script")
    parser.add_argument('--video_path', type=str, help='Path to the video file')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    path = args.video_path
    name = output_result(path)
    result1, result2 = results(name)
    if result1 == result1 == 'Anger' or result1 == 'Surprise':
        print(f'您上传的片段为{path},情感识别结果为愤怒')
    elif result1 == 'Fear' or result1 == 'Sadness' or result1 == 'Disgust':
        print(f'您上传的片段为{path},情感识别结果为悲伤')
    elif result1 == 'Trust':
        print(f'您上传的片段为{path},情感识别结果为中性')
    elif result1 == 'Anticipation' or result1 == 'Joy':
        print(f'您上传的片段为{path},情感识别结果为快乐')
