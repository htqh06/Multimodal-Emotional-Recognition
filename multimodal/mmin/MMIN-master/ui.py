from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog,
                             QTextEdit, QMessageBox, QGridLayout, QComboBox, QSlider)
from pydub import AudioSegment
from pydub.playback import play
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt

import make_comparE
from main import *
import threading
import os
import subprocess
import re
import json


class EmotionRecognitionApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.audio_segment = None
        self.audio_filename = None  # 保存当前音频文件名，无扩展名
        self.image_upload = False
        self.text_upload = False
        self.audio_upload = False
        self.video_filename = None
        self.test_filename = None
        self.video_filename = None
        self.text_filename = None
        self.video_filepath = None
        self.audio_filepath = None

    def initUI(self):
        self.setWindowTitle('多模态情感识别')
        self.setGeometry(100, 100, 1280, 720)  # 720p

        grid = QGridLayout()
        self.setLayout(grid)
        Bgrid = QVBoxLayout()

        # 创建介绍文字
        # label = QLabel("预训练好的模型:")
        # grid.addWidget(label, 0, 0)  # 放在第一行

        # 模型选择下拉框
        self.modelComboBox = QComboBox()
        self.load_model_directories()
        self.modelComboBox.setCurrentIndex(1)
        Bgrid.addWidget(self.modelComboBox)

        # 上传按钮
        self.btnUploadAudio = QPushButton('上传音频(.wav)')
        self.btnUploadAudio.clicked.connect(lambda: self.upload_file('audio'))
        Bgrid.addWidget(self.btnUploadAudio)

        self.btnUploadImage = QPushButton('上传图像(.mp4)')
        self.btnUploadImage.clicked.connect(lambda: self.upload_face())
        Bgrid.addWidget(self.btnUploadImage)

        self.btnUploadText = QPushButton('上传文本(.txt)')
        self.btnUploadText.clicked.connect(lambda: self.upload_file('text'))
        Bgrid.addWidget(self.btnUploadText)

        self.btnPlayAudio = QPushButton('播放音频')
        self.btnPlayAudio.clicked.connect(self.play_audio)
        Bgrid.addWidget(self.btnPlayAudio)

        # 播放视频按钮
        self.btnPlayVideo = QPushButton('播放视频')
        self.btnPlayVideo.clicked.connect(self.play_video)
        Bgrid.addWidget(self.btnPlayVideo)  # 放置于合适的位置

        self.btnStartTest = QPushButton('开始测试')
        self.btnStartTest.clicked.connect(self.run_model_test)
        Bgrid.addWidget(self.btnStartTest)

        self.textContent = QTextEdit()
        self.textContent.setReadOnly(True)
        self.textContent.setPlaceholderText("上传的文本内容将显示在这里")
        grid.addWidget(self.textContent, 5, 1, 4, 1)  # 中间列

        self.modelResults = QTextEdit()
        self.modelResults.setReadOnly(True)
        self.modelResults.setPlaceholderText("模型数据将显示在这里")
        grid.addWidget(self.modelResults, 0, 2, 4, 1)  # 右侧列

        self.textResults = QTextEdit()
        self.textResults.setReadOnly(True)
        self.textResults.setPlaceholderText("情感识别结果将显示在这里")
        grid.addWidget(self.textResults, 5, 2, 4, 1)  # 右侧列

        grid.addLayout(Bgrid, 0, 0, 8, 1)

        self.videoWidget = QVideoWidget()
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.videoWidget)
        grid.addWidget(self.videoWidget, 0, 1, 4, 1)

        # 创建进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)
        grid.addWidget(self.slider, 4, 1, 1, 1)

    def load_model_directories(self):
        model_path = 'checkpoints/'
        directories = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        self.modelComboBox.addItems(directories)

    def upload_file(self, filetype):
        if filetype == 'audio':
            file_filter = 'Audio files (*.wav)'
        elif filetype == 'text':
            file_filter = 'Text files (*.txt)'

        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '/', file_filter)
        if fname:
            if filetype == 'audio':
                self.audio_segment = AudioSegment.from_wav(fname)
                self.audio_filename = fname.split('/')[-1].split('.')[0]
                self.audio_filepath = fname
                self.modelResults.append(f'Audio file uploaded: {fname}')
                self.audio_upload = True
            elif filetype == 'text':
                self.extract_dialogue(fname)
                self.text_upload = True
                self.modelResults.append(f'{filetype.capitalize()} file uploaded: {fname}')

    def upload_face(self):
        self.image_upload = True
        filename, _ = QFileDialog.getOpenFileName(self, "打开视频", "", "视频文件 (*.mp4)")
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
        if filename:
            self.modelResults.append(f'video file uploaded: {filename}')
            self.video_filename = filename.split('/')[-1].split('.')[0]
            self.player.positionChanged.connect(self.position_changed)
            self.player.durationChanged.connect(self.duration_changed)
            self.video_filepath = filename

    def extract_dialogue(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith(self.audio_filename):
                    dialogue = line.strip()  # 分割并提取对话部分
                    self.textContent.setText(dialogue)
                    break

    def play_audio(self):
        if self.audio_segment:
            threading.Thread(target=self.play_audio_thread).start()
        else:
            QMessageBox.warning(self, 'Error', 'No audio file loaded.', QMessageBox.Ok)

    def play_audio_thread(self):
        play(self.audio_segment)

    def play_video(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()
         #   if self.audio_segment:
         #       threading.Thread(target=self.play_audio_thread).start()

    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)

    def set_position(self, position):
        self.player.setPosition(position)

    def run_model_test(self):
        if self.video_filename and not self.audio_upload and not self.text_upload and not self.video_filename.startswith('Ses'):
            os.chdir('D:/Graduation_project/multimodal/Emotion_rec/')
            self.textResults.clear()

            # 创建子进程
            process = subprocess.run(
                ['python', 'D:/Graduation_project/multimodal/Emotion_rec/main.py', '--video_path', self.video_filepath],
                stdout=subprocess.PIPE)

            # 读取输出并解码（text=True使得输出以字符串形式返回）
            output = process.stdout.decode('utf-8')
            print(output)
            split_keyword = "您"
            parts = output.split(split_keyword)
            clean_text = re.sub(r'\x1b\[.*?m', '', parts[1])
            self.textResults.append(clean_text)

        elif self.audio_filename and not self.audio_filename.startswith('Ses'):
            # 处理自定义上传音频
            os.chdir('D:/Graduation_project/multimodal/mmin/MMIN-master/')
            print('hello')
            self.textResults.clear()
            if self.audio_filename or self.video_filename or self.text_filename and self.modelComboBox.currentText():
                model_dir = self.modelComboBox.currentText()
                # 检查模态上传状态
                status = ""
                status += 'a' if self.audio_upload else 'z'
                status += 'v' if self.image_upload else 'z'
                status += 'l' if self.text_upload else 'z'
                if self.audio_filename:
                    self.test_filename = self.audio_filename

                with open('path_info.json', 'w') as f:
                    json.dump({'my_file_path': self.audio_filepath}, f)

                with open('name_info.json', 'w') as f:
                    json.dump({'my_file_path': self.audio_filename}, f)

                make_comparE.process_single_file(self.audio_filepath, self.audio_filename)

                try:
                    # 调用 mytest.py 并捕获输出
                    completed_process = subprocess.run(['python', 'pytest.py', self.test_filename, model_dir, status, self.audio_filepath],
                                                       stdout=subprocess.PIPE)
                    output = completed_process.stdout.decode("utf-8")
                    split_keyword = "注："
                    parts = output.split(split_keyword)
                    self.modelResults.append(parts[0])
                    print(output)
                    clean_text = re.sub(r'\x1b\[.*?m', '', parts[1])
                    self.textResults.append(clean_text)
                except subprocess.CalledProcessError as e:
                    QMessageBox.warning(self, 'Error', 'Model testing failed: ' + str(e), QMessageBox.Ok)
            else:
                QMessageBox.warning(self, 'Error', 'Please upload an audio file and select a ete_model.', QMessageBox.Ok)

        else:
            os.chdir('D:/Graduation_project/multimodal/mmin/MMIN-master/')
            self.textResults.clear()
            if self.audio_filename or self.video_filename or self.text_filename and self.modelComboBox.currentText():
                model_dir = self.modelComboBox.currentText()
                # 检查模态上传状态
                status = ""
                status += 'a' if self.audio_upload else 'z'
                status += 'v' if self.image_upload else 'z'
                status += 'l' if self.text_upload else 'z'
                if self.audio_filename:
                    self.test_filename = self.audio_filename
                elif self.video_filename:
                    self.test_filename = self.video_filename

                try:
                    # 调用 mytest.py 并捕获输出
                    completed_process = subprocess.run(['python', 'mytest.py', self.test_filename, model_dir, status],
                                                       stdout=subprocess.PIPE)
                    output = completed_process.stdout.decode("utf-8")
                    split_keyword = "注："
                    parts = output.split(split_keyword)
                    self.modelResults.append(parts[0])
                    print(output)
                    clean_text = re.sub(r'\x1b\[.*?m', '', parts[1])
                    self.textResults.append(clean_text)
                except subprocess.CalledProcessError as e:
                    QMessageBox.warning(self, 'Error', 'Model testing failed: ' + str(e), QMessageBox.Ok)
            else:
                QMessageBox.warning(self, 'Error', 'Please upload an audio file and select a ete_model.', QMessageBox.Ok)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EmotionRecognitionApp()
    ex.show()
    sys.exit(app.exec_())
