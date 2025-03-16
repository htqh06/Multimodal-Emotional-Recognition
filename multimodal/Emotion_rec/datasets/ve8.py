import torch
import torch.utils.data as data

from torchvision import get_image_backend

from PIL import Image

import json
import os
import functools
import librosa
import numpy as np


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        return float(input_file.read().rstrip('\n\r'))


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []
    for key, value in data['database'].items():
        if value['subset'] == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])
    return video_names, annotations


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
        assert os.path.exists(image_path), "image does not exists"
        video.append(image_loader(image_path))
    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def preprocess_audio(audio_path):
    "Extract audio features from an audio file"
    y, sr = librosa.load(audio_path, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
    return mfccs


class VE8Dataset(data.Dataset):
    def __init__(self,
                 video_path,
                 audio_path,
                 annotation_path,
                 subset,
                 fps=30,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 get_loader=get_default_video_loader,
                 ind=None,
                 need_audio=True):
        # print("index",ind)
        if ind == None:
            self.data, self.class_names = make_dataset(
                video_root_path=video_path,
                annotation_path=annotation_path,
                audio_root_path=audio_path,
                subset=subset,
                fps=fps,
                need_audio=need_audio
            )
        else:
            self.data, self.class_names = make_dataset2(
                video_root_path=video_path,
                annotation_path=annotation_path,
                audio_root_path=audio_path,
                ind=ind,
                subset=subset,
                fps=fps,
                need_audio=need_audio
            )
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.fps = fps
        self.ORIGINAL_FPS = 30
        self.need_audio = need_audio

    def __getitem__(self, index):
        data_item = self.data[index]
        video_path = data_item['video']
        frame_indices = data_item['frame_indices']
        snippets_frame_idx = self.temporal_transform(frame_indices)

        if self.need_audio:
            timeseries_length = 4096
            audio_path = data_item['audio']
            feature = preprocess_audio(audio_path).T
            k = timeseries_length // feature.shape[0] + 1
            feature = np.tile(feature, reps=(k, 1))
            audios = feature[:timeseries_length, :]
            audios = torch.FloatTensor(audios)
        else:
            audios = []

        snippets = []
        for snippet_frame_idx in snippets_frame_idx:
            snippet = self.loader(video_path, snippet_frame_idx)
            snippets.append(snippet)

        self.spatial_transform.randomize_parameters()
        snippets_transformed = []
        for snippet in snippets:
            snippet = [self.spatial_transform(img) for img in snippet]
            snippet = torch.stack(snippet, 0).permute(1, 0, 2, 3)
            snippets_transformed.append(snippet)
        snippets = snippets_transformed
        snippets = torch.stack(snippets, 0)

        target = self.target_transform(data_item)
        visualization_item = [data_item['video_id']]

        return snippets, target, audios, visualization_item

    def __len__(self):
        return len(self.data)

ORIGINAL_FPS = 30
def load_and_validate_paths(video_path, audio_path=None):
    assert os.path.exists(video_path), f"{video_path} does not exist"
    if audio_path:
        assert os.path.exists(audio_path), f"{audio_path} does not exist"

def create_sample(video_path, audio_path, n_frames, annotations, class_to_idx, fps):
    begin_t, end_t = 1, n_frames
    frame_indices = list(range(begin_t, end_t + 1, ORIGINAL_FPS // fps))
    return {
        'video': video_path,
        'audio': audio_path,
        'frame_indices': frame_indices,
        'label': class_to_idx[annotations['label']],
        'video_id': os.path.basename(video_path).split('.')[0]
    }

def make_dataset(video_root_path, annotation_path, audio_root_path, subset, fps=30, need_audio=True):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)

    dataset = []
    for video_name, annotation in zip(video_names, annotations):
        video_path = os.path.join(video_root_path, video_name.split("/")[1])
        audio_path = os.path.join(audio_root_path, "Joy.mp3") if need_audio else None
        n_frames = int(load_value_file(os.path.join(video_path, 'n_frames')))
        if n_frames > 0:
            load_and_validate_paths(video_path, audio_path)
            sample = create_sample(video_path, audio_path, n_frames, annotation, class_to_idx, fps)
            dataset.append(sample)

    return dataset, {v: k for k, v in class_to_idx.items()}

def make_dataset2(video_root_path, annotation_path, audio_root_path, ind, subset, fps=30, need_audio=True):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)

    dataset = []
    for video_name, annotation in zip(video_names, annotations):
        video_path = os.path.join(video_root_path, video_name.split("/")[1])
        audio_path = os.path.join(audio_root_path, f"output{ind}.mp3") if need_audio else None
        n_frames = int(load_value_file(os.path.join(video_path, 'n_frames')))
        if n_frames > 0:
            load_and_validate_paths(video_path, audio_path)
            sample = create_sample(video_path, audio_path, n_frames, annotation, class_to_idx, fps)
            dataset.append(sample)

    return dataset, {v: k for k, v in class_to_idx.items()}
# Assumed existence of accimage_loader if using accimage