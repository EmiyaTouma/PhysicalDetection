# 测试变心率

# -*- coding: utf-8 -*-
#########基本库引用##########
import cv2
import numpy as np
from dlib import get_frontal_face_detector, shape_predictor
import os
from time import time, strftime
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from statistics import mode
from pywt import Wavelet, upcoef, wavedec
from hrvanalysis import get_frequency_domain_features
from hrvanalysis import get_poincare_plot_features
from scipy import interpolate
from joblib import load as joblib_load
from PIL import Image
import EasyPySpin
from facenet import Facenet

#########qt程序库引用##########
import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from qimage2ndarray import array2qimage

########进程程序库引用###########
from multiprocessing import Process, Queue
from collections import deque

#############界面#################
from cdjm import *
from data import *

MIN_FRAMES = 100  # 计算心率之前所需的最小帧数。 较高的值速度较慢，但更准确。
MIN_FRAMES_HRV = 300
FPS = 10
TOTALTIME = 600  # 总的记录时长，方便测试时进行修改，正常为300s，测试时可改为30s/60s


######################################### ResNet表情检测 ###################################################
class ResNet(nn.Module):
    def __init__(self, *args):
        super(ResNet, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


# 残差神经网络
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


########################################################################################################

####################### 图像处理进程 ############################
# 计算
class Heartbeat:
    def __init__(self, frames, q, con1, name):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 人脸识别
        self.detector = get_frontal_face_detector()
        # 关键点检测
        self.predictor = shape_predictor('model/shape_predictor_68_face_landmarks.dat')
        # 表情识别
        self.emotion_classifier = torch.load('model/model_resnet1.pkl', map_location=device)
        # SVM分类
        self.svmRF = joblib_load('model/svmRF.pkl')
        # 人脸检测网络
        self.model_face = Facenet()

        # 设置双端队列，并设置最大窗口
        self.bpm_avg_values = deque(maxlen=15 * FPS)  # 最多保存30s数据用于bpm计算
        self.hrv_avg_values = deque(maxlen=35 * FPS)  # 最多保存180s数据用于hrv计算
        self.br_avg_values = deque(maxlen=15 * FPS)  # 最多保存30s数据用于br计算
        self.SpO2_avg_values_R = deque(maxlen=15 * FPS)  # 最多保存30s数据用于SpO2计算
        self.SpO2_avg_values_B = deque(maxlen=15 * FPS)
        self.emotion_window = deque(maxlen=35 * FPS)  # 最多保存180s表情数据
        self.graph = deque([0 for _ in range(10 * FPS)], maxlen=10 * FPS)
        self.lastbxz = deque([0 for _ in range(2)], maxlen=2)
        self.frameNum = 0  # 记录读取帧数，到达300*FPS帧结束
        self.time1 = 0  # 运算时间
        self.countframe = 0
        self.jishu = 0
        self.nameNum = 0
        self.nameError = 0
        self.ori_sig(frames, q, con1, name)  # frames传入图像，q传出结果，con1传入程序运行状态

    def ori_sig(self, frames, q, con1, name):
        print('初始化信号')

        # 表情相关参数
        emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'calm'}
        emotion_mode = '--'  # emo mode
        cxmode = 1  # 当前程序运行模式 1检测正常；2检测完成；3检测异常
        bpm = 0  # bpm
        lastbpm = 0
        br = 0
        # lastbr = 0
        SpO2 = 0
        # lastSpO2 = 0
        rr = 0
        hrvlist = [0 for _ in range(2)]  # hrv参数记录
        lasthrvlist = [0 for _ in range(2)]
        result = 0  # 最终result
        yunxing = 0  # 控制程序是否运行 1运行；0重置
        graph1 = np.ones((200, 800, 3), np.uint8) * 255
        graph1 = self.draw_graph(self.graph, 1700, 200)
        while True:
            if not con1.empty():
                yunxing = con1.get()
            if yunxing == 1:
                while True:
                    if not con1.empty():
                        yunxing = con1.get()
                        if yunxing == 0:
                            break
                    if not frames.empty():
                        self.time1 += 1
                        self.frameNum += 1
                        frame = frames.get()
                        # 人脸识别
                        faces = self.detector(frame, 0)

                        if faces:
                            cxmode = 1
                            self.countframe += 1
                            self.jishu = 0
                            # 获取人脸位置
                            x, y = faces[0].left(), faces[0].top()
                            width, h = faces[0].right() - x, faces[0].bottom() - y

                            # 获取人脸灰度图
                            face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + width]
                            # 改变维度
                            try:
                                face = cv2.resize(face, (48, 48))
                            except:
                                continue
                            # 扩充维度，shape变为(1,48,48,1)将（1,48,48,1）转换成为(1,1,48,48)
                            face = np.expand_dims(face, 0)
                            face = np.expand_dims(face, 0)
                            # 归一化
                            face = face / 255.0
                            face = torch.from_numpy(face)
                            face = face.float().requires_grad_(False)
                            # 转成cuda数据
                            face = face.cuda()
                            # 分类表情
                            emotion_arg = np.argmax(self.emotion_classifier.forward(face).cpu().detach().numpy())
                            emotion = emotion_labels[emotion_arg]
                            self.emotion_window.append(emotion)

                            # 关键点检测
                            face_points = self.predictor(frame, faces[0])
                            # 获取ROI，roi为1*3的list，分别代表RGB
                            roi_avg, roi_hrv, roi_br, roi_SpO2_R, roi_SpO2_B = self.get_roi_avg(frame, face_points)
                            self.bpm_avg_values.append(roi_avg)
                            self.hrv_avg_values.append(roi_hrv)
                            self.br_avg_values.append(roi_br)
                            self.SpO2_avg_values_R.append(roi_SpO2_R)
                            self.SpO2_avg_values_B.append(roi_SpO2_B)

                            # 一秒计算一次
                            if self.countframe == FPS:
                                self.countframe = 0
                                # emotion
                                try:
                                    emotion_mode = mode(self.emotion_window)
                                except:
                                    pass

                                # 检测姓名
                                if not self.nameNum:
                                    self.detect_name(frame, x, y, width, h, name)

                                # bpm
                                if len(self.bpm_avg_values) > MIN_FRAMES:  # 超过可以计算心率所需最小帧数
                                    # 降噪滤波
                                    filtered = self.filter_signal_data(FPS)
                                    # 计算心率
                                    bpm = self.compute_hr(filtered, FPS)
                                    if lastbpm != 0:
                                        bpm = int(lastbpm * 0.9 + bpm * 0.1)
                                    lastbpm = bpm

                                    # br
                                    br = self.compute_br(self.xiaobo(self.btws(self.br_avg_values, FPS)), FPS)
                                    # SpO2
                                    SpO2 = self.compute_SpO2(self.btws(self.SpO2_avg_values_R, FPS),
                                                             self.btws(self.SpO2_avg_values_B, FPS))

                                # HRV
                                if len(self.hrv_avg_values) > MIN_FRAMES_HRV:  # 超过可以计算hrv所需最小帧数
                                    emotionresult = self.faceemotion()
                                    # 计算心率变异性
                                    # 清理信号数据
                                    hrvfiltered = self.hrvyuchuli()
                                    # 开始计算心率变异性
                                    hrvlist = self.compute_hrv(hrvfiltered)
                                    hrvscore = self.fenlei(bpm, hrvlist)
                                    result = self.juece(emotionresult, hrvscore)
                                    for i in range(len(lasthrvlist)):
                                        if lasthrvlist[i] != 0:
                                            hrvlist[i] = round(lasthrvlist[i] * 0.9 + hrvlist[i] * 0.1, 2)
                                        lasthrvlist[i] = hrvlist[i]

                            boxingzhi = self.calculate(bpm, self.lastbxz)
                            self.lastbxz.append(boxingzhi)
                            self.graph.append(boxingzhi)
                            graph1 = self.draw_graph(self.graph, 1700, 200)

                        # 没检测到人脸，5s内可继续检测，超出5s重新检测
                        else:
                            cxmode = 3
                            self.countframe = 0
                            self.jishu += 1
                            self.graph.append(0)
                            graph1 = self.draw_graph(self.graph, 1700, 200)
                            if self.jishu >= 5 * FPS:
                                self.time1 = 0
                                self.frameNum = 0
                                self.nameNum = 0
                                self.bpm_avg_values.clear()
                                self.hrv_avg_values.clear()
                                self.emotion_window.clear()
                                self.br_avg_values.clear()
                                self.SpO2_avg_values_R.clear()
                                self.SpO2_avg_values_B.clear()
                                self.graph = deque([0 for _ in range(10 * FPS)], maxlen=10 * FPS)
                                self.lastbxz = deque([0 for _ in range(2)], maxlen=2)
                                lastbpm = 0
                                # lastbr = 0
                                # lastSpO2 = 0
                                lasthrvlist = [0 for _ in range(2)]

                        if q.empty():
                            if bpm != 0:
                                rr = int(60*1000/bpm)
                            q.put([np.array(frame), emotion_mode, bpm, rr, cxmode, self.time1, br, SpO2,
                                   np.array(graph1)])

                    elif self.frameNum == TOTALTIME * FPS:
                        break

                # 数据采集结束，结果计算部分
                if yunxing == 1:
                    print('开始计算')
                    # bpm, hrvlist, br, SpO2 = self.computeresult()
                    emotionresult = self.faceemotion()
                    emotion_mode = mode(self.emotion_window)
                    hrvscore = self.fenlei(bpm, hrvlist)
                    result = self.juece(emotionresult, hrvscore)
                    cxmode = 2
                    if bpm != 0:
                        rr = int(60*1000/bpm)
                    q.put([np.array(frame), emotion_mode, bpm, rr, cxmode, self.time1, br, SpO2])
                    self.bpm_avg_values.clear()
                    self.hrv_avg_values.clear()
                    self.emotion_window.clear()
                    self.br_avg_values.clear()
                    self.SpO2_avg_values_R.clear()
                    self.SpO2_avg_values_B.clear()
                    self.graph = deque([0 for _ in range(10 * FPS)], maxlen=10 * FPS)
                    self.lastbxz = deque([0 for _ in range(2)], maxlen=2)
                    yunxing = 0
                    self.time1 = 0
                    self.nameNum = 0
                    self.nameError = 0
                    print('bpm:', bpm, '\nhrvlist:', hrvlist, '\nemotion:', emotionresult, '\nbr:', br, '\nSpO2:', SpO2)
                    print('得分：', result)

            # 重置数据
            elif yunxing == 0:
                self.time1 = 0
                self.frameNum = 0
                self.bpm_avg_values.clear()
                self.hrv_avg_values.clear()
                self.emotion_window.clear()
                self.br_avg_values.clear()
                self.SpO2_avg_values_R.clear()
                self.SpO2_avg_values_B.clear()
                self.graph = deque([0 for _ in range(10 * FPS)], maxlen=10 * FPS)
                self.lastbxz = deque([0 for _ in range(2)], maxlen=2)
                self.nameNum = 0
                self.nameError = 0
                emotion_mode = '--'
                bpm = 0
                result = 0
                rr = 0
                br = 0
                SpO2 = 0
                graph1 = self.draw_graph(self.graph, 1700, 200)
                while not q.empty():
                    q.get()

    # 计算名字
    def detect_name(self, last_frame, x, y, width, h, name):
        # 获取人脸图
        face = last_frame[y:y + h, x:x + width]
        img_1 = Image.fromarray(np.uint8(face))  # 将图片从numpy类型转为PIL类型
        path = "img"
        filelist = os.listdir(path)  # 遍历该文件夹下所有的文件（包括文件夹）
        probability = 2
        for image_id in filelist:
            tempPath = "img/"
            image_path = tempPath + image_id
            # image_1 = cv2.imread(image_path)
            image_1 = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR)
            faces = self.detector(image_1, 0)
            if faces:
                # 获取人脸位置
                x, y = faces[0].left(), faces[0].top()
                width, h = faces[0].right() - x, faces[0].bottom() - y

            # 获取人脸图
            face = image_1[y:y + h, x:x + width]
            img_2 = Image.fromarray(np.uint8(face))  # 将图片从numpy类型转为PIL类型
            probability = self.model_face.detect_image(img_1, img_2)  # 进行人脸识别
            print(probability)
            if probability < 1:  # 若置信度小于1，即找到目标
                self.nameNum = 1
                dataset_path = image_id.split('.')[0]
                name.put(dataset_path)
                dataset_path = "dataset/" + image_id
                # self.SetHistory()
                break
        if probability > 1:
            self.nameError = self.nameError + 1
            if self.nameError > 10:
                self.time1 = 0
                self.frameNum = 0
                self.nameNum = 0
                self.nameError = 0
                self.bpm_avg_values.clear()
                self.hrv_avg_values.clear()
                self.emotion_window.clear()
                self.br_avg_values.clear()
                self.SpO2_avg_values_R.clear()
                self.SpO2_avg_values_B.clear()
                self.graph = deque([0 for _ in range(10 * FPS)], maxlen=10 * FPS)
                self.lastbxz = deque([0 for _ in range(2)], maxlen=2)
                self.msg3(last_frame)

    def msg3(self, last_frame):
        app = QApplication(sys.argv)
        temp_box = QWidget()
        temp_box.setWindowOpacity(0)
        temp_box.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        temp_box.show()
        input_box = QInputDialog()
        value, ok = input_box.getText(temp_box, "请输入您的姓名", "请在下方文本框中输入您的姓名:", QLineEdit.Normal, "")
        path = 'img/' + value + '.jpg'
        img = Image.fromarray(np.uint8(last_frame))
        img.save(path)
        # msgBox = QMessageBox()
        # msgBox.setIcon(QMessageBox.Warning)
        # msgBox.setText("警告")
        # msgBox.setInformativeText("请在数据库中补充人脸图片")
        # msgBox.setWindowTitle("警告")
        # msgBox.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)
        # msgBox.exec_()
        # sys.exit(app.exec_())

    # Returns maximum absolute value from a list
    def get_max_abs(self, signalvalues):
        return max(max(signalvalues), -min(signalvalues))

    # Draws the heart rate graph in the GUI window.
    def draw_graph(self, signal_values, graph_width, graph_height):
        graph = np.ones((graph_height, graph_width, 3), np.uint8) * 255
        scale_factor_x = float(graph_width) / 100
        # Automatically rescale vertically based on the value with largest absolute value
        max_abs = self.get_max_abs(signal_values)
        # scale_factor_y = (float(graph_height) / 2.0) / max_abs
        scale_factor_y = int((float(graph_height) / 2.0) / 2)
        midpoint_y = graph_height / 2
        for i in range(0, len(signal_values) - 1):
            curr_x = int(i * scale_factor_x)
            curr_y = int(midpoint_y + signal_values[i] * scale_factor_y)
            next_x = int((i + 1) * scale_factor_x)
            next_y = int(midpoint_y + signal_values[i + 1] * scale_factor_y)
            cv2.line(graph, (curr_x, curr_y), (next_x, next_y), color=(255, 0, 0), thickness=2)
        return graph

        # 计算曲线

    def calculate(self, bpm, lastbxz):
        w = bpm * np.pi / 30
        lastfai = np.arcsin(lastbxz[1])
        if lastbxz[0] == 0:
            pass
        else:
            if (lastbxz[0] > lastbxz[1]):
                lastfai = np.pi - lastfai
        bxz = np.sin(w / FPS + lastfai)
        return bxz

    # 全部结束后计算结果
    def computeresult(self):
        # 降噪滤波
        filtered = self.filter_signal_data(FPS)
        # 计算心率
        bpm = self.compute_hr(filtered, FPS)
        br = self.compute_br(self.xiaobo(self.btws(self.br_avg_values, FPS)), FPS)
        SpO2 = self.compute_SpO2(self.btws(self.SpO2_avg_values_R, FPS), self.btws(self.SpO2_avg_values_B, FPS))

        # 计算心率变异性
        # 清理信号数据
        hrvfiltered = self.hrvyuchuli()
        # 开始计算心率变异性
        hrvlist = self.compute_hrv(hrvfiltered)
        return bpm, hrvlist, br, SpO2

    # 统计人脸表情结果
    def faceemotion(self):
        emotion = [0, 0, 0]
        chang = len(self.emotion_window)
        for i in self.emotion_window:
            if i == 'angry' or i == 'disgust' or i == 'fear' or i == 'sad':
                emotion[0] += 1
            elif i == 'happy' or i == 'surprise':
                emotion[1] += 1
            elif i == 'calm':
                emotion[2] += 1
        for i in range(3):
            emotion[i] = round(emotion[i] / chang, 2)
        return emotion

    # 回归分析
    def fenlei(self, bpm, hrvlist):
        result = [0, 0, 0]
        result[0] = bpm
        result[1] = hrvlist[0]
        result[2] = hrvlist[1]
        result1 = [result]
        score = self.svmRF.predict(result1)
        return score

    # 决策融合
    def juece(self, emotionresult, hrvscore):
        if hrvscore == 0:
            scorehrv = 0
        elif hrvscore == 1:
            scorehrv = 50
        scoreemo = 25 + 25 * emotionresult[0] - 25 * emotionresult[1]
        score = scorehrv + scoreemo
        return score

    # 获取ROI
    def get_roi_avg(self, frame, face_points):
        # 获取鼻子位置
        points = np.zeros((len(face_points.parts()), 2))
        for i, part in enumerate(face_points.parts()):
            points[i] = (part.x, part.y)
        min_x = int(points[37, 0])
        min_y = int(points[29, 1])
        max_x = int(points[46, 0])
        max_y = int(points[34, 1])
        left = min_x
        right = max_x
        top = int(min_y + (min_y * 0.02))
        bottom = int(max_y + (max_y * 0.02))
        # 切出感兴趣区域（ROI）并平均它们
        nose_roi = frame[top:bottom, left:right]
        # [:, :, 2]，三维数列的索引，表示前两个维度不限制，第三个维度只取第2.
        # 三个维度分别为BGR
        return [np.mean(nose_roi[:, :, 2]), np.mean(nose_roi[:, :, 1]), np.mean(nose_roi[:, :, 0])], np.mean(
            nose_roi[:, :, 1]), np.mean(
            nose_roi[:, :, 1]), np.mean(nose_roi[:, :, 2]), np.mean(nose_roi[:, :, 0])

    # 降噪滤波
    def filter_signal_data(self, fps):
        values = np.mat(self.bpm_avg_values).transpose()
        # CDF滤波
        def CDF(C):
            B = [7, 40]
            di = np.mat(np.diag(np.average(C, axis=1).transpose().tolist()[0]))
            C1 = di.I * C - 1
            F = np.mat(np.fft.fft(C1))
            S = np.mat([-1, 2, -1]) / np.sqrt(6) * F
            W = np.divide(np.multiply(S, np.conj(S)), np.sum(np.multiply(F, np.conj(F)), axis=0))
            W[:, 0:B[0] - 1] = 0
            W[:, B[1] + 1:] = 0
            F1 = np.multiply(F, np.r_[W, W, W])
            C_new = di * (np.mat(np.fft.ifft(F1)) + 1).real
            return C_new

        # chrom滤波
        def chrom(sub_RGB):
            RGBmean = np.average(sub_RGB, axis=1)
            for i in range(3):
                sub_RGB[i, :] = sub_RGB[i, :] / RGBmean[i, 0]
            Xs = 3 * sub_RGB[0, :] - 2 * sub_RGB[1, :]
            Ys = 1.5 * sub_RGB[0, :] + sub_RGB[1, :] - 1.5 * sub_RGB[2, :]
            # 带通滤波
            a, b = signal.butter(8, [0.7 / fps, 4 / fps], 'bandpass')
            Xf = signal.filtfilt(a, b, Xs)
            Yf = signal.filtfilt(a, b, Ys)
            a = np.std(Xf, ddof=1) / np.std(Yf, ddof=1)
            return Xf - a * Yf

        # 移动平均
        def sliding_window_demean(signal_values, num_windows):
            window_size = int(round(len(signal_values) / num_windows))
            demeaned = np.zeros(signal_values.shape)
            for i in range(0, len(signal_values), window_size):
                if i + window_size > len(signal_values):
                    window_size = len(signal_values) - i
                curr_slice = signal_values[i: i + window_size]
                demeaned[i:i + window_size] = curr_slice - np.mean(curr_slice)
            return demeaned

        # 去趋势
        def detrend(x):
            detrended = signal.detrend(x, type='linear')
            detrended1 = detrended.flatten()
            demeaned = sliding_window_demean(detrended1, 15)  # 这个地方的15要不要改为1.5*FPS有待考究
            return demeaned

        return detrend(chrom(CDF(values)))

    # 计算心率
    def compute_hr(self, filtered_values, fps):
        IPPG = filtered_values.tolist()
        N = len(IPPG)
        y = np.fft.fft(IPPG, N)
        mag = np.abs(y)
        mag = ((mag - np.mean(mag)) / np.std(mag, ddof=1)).transpose().tolist()
        minbpm = int(np.ceil(1 * N / fps))
        maxbpm = int(np.floor(2.5 * N / fps))
        maxbr = int(np.floor(0.7 * N / fps))
        v0 = max(mag[0:maxbr])
        i0 = mag.index(v0)
        v1 = max(mag[minbpm:maxbpm])
        i1 = mag.index(v1, minbpm, maxbpm)
        bpm = int(i1 * fps / N * 60)
        # 返回心率和呼吸率
        return int(bpm)

    # 计算呼吸率，SpO2
    # 巴特沃斯
    def btws(self, x, fps):
        a, b = signal.butter(8, [0.4 / fps, 1.6 / fps], 'bandpass')
        Xf = signal.filtfilt(a, b, x)
        return Xf

    # 小波
    def xiaobo(self, x):
        data = x
        w = Wavelet('db12')
        [ca7, cd7, cd6, cd5, cd4, cd3, cd2, cd1] = wavedec(data, w, level=7)
        n = len(data)
        datarec = upcoef('d', cd7, 'db12', level=7, take=n) + upcoef('d', cd6, 'db12', level=6,
                                                                               take=n) + upcoef('d', cd5, 'db12',
                                                                                                     level=5, take=n)
        return datarec

    # 呼吸率
    def compute_br(self, filtered_values, fps):
        IPPG = filtered_values.tolist()
        N = len(IPPG)
        X_peak = signal.find_peaks(IPPG)[0]
        number = len(X_peak)
        br = int(number * fps / N * 60)
        # 返回呼吸率
        return int(br)

    def cptave_SpO2(self, x, mode):
        sumx = 0
        y = []
        X = x * mode
        X_peak = signal.find_peaks(X)[0]
        length = len(X_peak)
        for i in X_peak:
            sumx = sumx + x[i]
        ave = sumx / length
        return ave

    # 计算SpO2
    def compute_SpO2(self, R, B):
        a = 93.8854
        b = 5.2264
        I_R = (self.cptave_SpO2(R, 1) - self.cptave_SpO2(R, -1)) / (np.mean(R) + 0.5)
        I_B = (self.cptave_SpO2(B, 1) - self.cptave_SpO2(B, -1)) / (np.mean(B) + 0.5)
        SpO2 = a + b * (I_R / I_B)
        if SpO2 < 96:
            SpO2 = 96
        if SpO2 > 100:
            SpO2 = 100
        # 返回SpO2
        return int(SpO2)

    # hrv预处理
    def hrvyuchuli(self):
        # 插值至100帧
        def chazhi(b0):
            blen = len(b0)
            x = np.linspace(0, blen - 1, blen)
            f = interpolate.interp1d(x, b0, kind='slinear')
            xnew = np.linspace(0, blen - 1, 10 * blen - 9)  # 这个地方的插值目前是按照10帧计算的插值100帧，如何变成变动的还需要考虑
            return f(xnew)

        # 创建巴特沃斯过滤器
        def butterworth_filter(data, low, high, sample_rate, order=5):
            nyquist_rate = sample_rate * 0.5
            low /= nyquist_rate
            high /= nyquist_rate
            b, a = signal.butter(order, [low, high], btype='band')
            return signal.lfilter(b, a, data)

        # 移动平均
        def sliding_window_demean(signal_values, num_windows):
            window_size = int(round(len(signal_values) / num_windows))
            demeaned = np.zeros(signal_values.shape)
            for i in range(0, len(signal_values), window_size):
                if i + window_size > len(signal_values):
                    window_size = len(signal_values) - i
                curr_slice = signal_values[i: i + window_size]
                demeaned[i:i + window_size] = curr_slice - np.mean(curr_slice)
            return demeaned

        # 去趋势
        def detrend(x):
            detrended = signal.detrend(x, type='linear')
            detrended1 = detrended.flatten()
            return detrended1

        # 小波
        def xiaobo(x):
            data = x
            w = Wavelet('db12')
            [ca7, cd7, cd6, cd5, cd4, cd3, cd2, cd1] = wavedec(data, w, level=7)
            n = len(data)
            datarec = upcoef('d', cd7, 'db12', level=7, take=n) + upcoef('d', cd6, 'db12', level=6,
                                                                                   take=n) + upcoef('d', cd5,
                                                                                                         'db12',
                                                                                                         level=5,
                                                                                                         take=n)
            return datarec

        # 此处100为插值100帧，如果后面插值结果发生变化需要进行调整，看如何适应变化的FPS
        return xiaobo(
            butterworth_filter(sliding_window_demean(detrend(chazhi(self.hrv_avg_values)), 25), 0.83, 3.33, 100,
                               order=5))

    # 计算心率变异性
    def compute_hrv(self, hrvfiltered):
        hrvlist = [0 for _ in range(2)]
        peaks, _ = signal.find_peaks(hrvfiltered, height=0.1, distance=40, width=15)
        # 计算峰值之间间隔
        dis = []
        for i in range(len(peaks) - 1):
            dis.append((peaks[i + 1] - peaks[i]) * 1000 / 100)  # 此处100为插值帧数100帧，1000为转换为ms单位
        for i in range(1, len(dis) - 2):
            if (dis[i] <= 500 or dis[i] >= 1200):
                dis[i] = (dis[i - 1] + dis[i + 1]) / 2
                if dis[i] > 1200:
                    dis[i] = 1200
                elif dis[i] < 500:
                    dis[i] = 500
        frequency_domain_features = get_frequency_domain_features(dis)
        poincare_plot_features = get_poincare_plot_features(dis)
        hrvlist[0] = frequency_domain_features['lf_hf_ratio']
        hrvlist[1] = poincare_plot_features['ratio_sd2_sd1']
        return hrvlist


################################## 图像获取进程 ##############################
# 获取5min图像
class Td:
    def __init__(self, frames, con, img_lists):
        self.cap = None
        self.frameNum = 0
        self.data_get(frames, con, img_lists)  # q传出图像，con传入状态

    def data_get(self, frames, con, img_lists):
        yunxing = 0
        while True:
            if not con.empty():
                yunxing = con.get()
            if yunxing == 1:
                # self.cap = cv2.VideoCapture('ceshi.mp4')
                self.cap = EasyPySpin.VideoCapture(0)
                ret, last_frame = self.cap.read()
                last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BayerBG2BGR)
                last_time1 = time()

                # 测试时可调整为30s采样，即30 * FPS
                while self.frameNum < TOTALTIME * FPS:
                    if not con.empty():
                        yunxing = con.get()
                        if yunxing == 0:
                            break
                    self.frameNum += 1
                    while True:
                        tmp = time()
                        # 此处99为根据10FPS计算得出，如修改FPS则应修改此处值为(1000/FPS - 1)
                        if int((tmp - last_time1) * 1000) == 99:
                            ret, frame = self.cap.read()
                            frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
                            if ret:
                                frames.put(frame)
                                img_lists.put(np.array(frame))
                                last_frame = frame
                                last_time1 = time()
                                break
                        elif int((tmp - last_time1) * 1000) > 99:
                            frames.put(last_frame)
                            img_lists.put(np.array(last_frame))
                            last_time1 = time()
                            break

                    last_time1 = tmp
            elif yunxing == 0:
                if self.cap is not None:
                    self.cap.release()
                self.frameNum = 0


###############################历史数据显示框体####################################
class Disp_History(QWidget, Ui_History):
    def __init__(self, historys):
        super(Disp_History, self).__init__()
        self.setupUi(self)
        self.yushe(historys)

    def yushe(self, historys):
        for i in range(len(historys)):
            history = historys[i]
            row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row)
            items = history.split("  ")
            for j in range(len(items)):
                if j == 0:
                    item = QTableWidgetItem(items[j][:-1])
                elif j == 5:
                    item = QTableWidgetItem(items[j][5:])
                else:
                    item = QTableWidgetItem(items[j][2:])
                self.tableWidget.setItem(row, j, item)


################################# 图像显示进程 ###################################3
# 界面显示
class mwindow(QWidget, Ui_jiemian):
    def __init__(self):
        super(mwindow, self).__init__()
        self.textNum = 0
        self.timeControl = 0
        self.m = None
        self.setupUi(self)
        self.CallBackFunctions()  # 定义回调函数
        self.model_face = Facenet()
        self.yushe()
        self.Timer = QTimer()  # for 实时输出
        self.Timer.timeout.connect(self.TimerOutFun)  # 主程序

    # 预设
    def yushe(self):
        self.output = [0, 0, 0, 0, 0]
        self.detect_Name.setText('--')  # 检测人姓名
        self.emo.setText('--')  # 情绪
        self.bpm.setText('--')  # 心率
        self.br.setText('--')  # br
        self.SpO2.setText('--')  # spo2
        self.score.setText('--')  # 得分
        # self.tishi.setText('按下开始按钮进行检测')
        self.xianshi.setText('检测时请面朝摄像头')
        pixmap = QPixmap('face.png')
        scaredPixmap = pixmap.scaled(450, 300, aspectRatioMode=Qt.KeepAspectRatio)
        self.frame.setPixmap(scaredPixmap)
        self.frame.show()

    def detect_name(self):
        # self.cap = cv2.VideoCapture('ceshi.mp4')
        self.cap = EasyPySpin.VideoCapture(0)
        ret, last_frame = self.cap.read()
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BayerBG2BGR)
        self.detector = get_frontal_face_detector()
        faces = self.detector(last_frame, 0)
        while len(faces) == 0:
            ret, last_frame = self.cap.read()
            last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BayerBG2BGR)
            faces = self.detector(last_frame, 0)
        self.cap.release()
        if faces:
            # 获取人脸位置
            x, y = faces[0].left(), faces[0].top()
            w, h = faces[0].right() - x, faces[0].bottom() - y

        # 获取人脸图
        face = last_frame[y:y + h, x:x + w]
        img_1 = Image.fromarray(np.uint8(face))  # 将图片从numpy类型转为PIL类型
        path = "img"
        filelist = os.listdir(path)  # 遍历该文件夹下所有的文件（包括文件夹）
        probability = 2
        for image_id in filelist:
            tempPath = "img/"
            image_path = tempPath + image_id
            # image_1 = cv2.imread(image_path)
            image_1 = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR)
            faces = self.detector(image_1, 0)
            if faces:
                # 获取人脸位置
                x, y = faces[0].left(), faces[0].top()
                w, h = faces[0].right() - x, faces[0].bottom() - y

            # 获取人脸图
            face = image_1[y:y + h, x:x + w]
            img_2 = Image.fromarray(np.uint8(face))  # 将图片从numpy类型转为PIL类型
            probability = self.model_face.detect_image(img_1, img_2)  # 进行人脸识别
            print(probability)
            if probability < 1:  # 若置信度小于1，即找到目标
                dataset_path = image_id.split('.')[0]
                self.detect_Name.setText(dataset_path)
                dataset_path = "dataset/" + image_id
                # self.SetHistory()
                break
        if probability > 1:
            self.msg3()
            sys.exit()

    def msg3(self):
        QMessageBox.warning(self, "标题", "请在数据库中补充人脸图片", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    def SetHistory(self):
        dataset_path = "dataset/" + self.detect_Name.text() + ".txt"
        file = open(dataset_path, 'a+')  # 以读写模式打开
        file.seek(0)  # 移动指针到开头
        history = file.readlines()
        print(history)
        if len(history) >= 2:
            self.history_1.setText(history[-1])
            self.history_2.setText(history[-2])
        elif len(history) == 1:
            self.history_1.setText(history[-1])
            self.history_2.setText("--")
        elif len(history) == 0:
            self.history_1.setText("--")
            self.history_2.setText("--")
        file.close()

    # 回调函数表
    def CallBackFunctions(self):
        self.kaishi.clicked.connect(self.StartCamera)
        self.jieshu.clicked.connect(self.StopCamera)
        # self.history.clicked.connect(self.on_history_clicked())

    def TimerOutFun(self):
        if not q.empty():
            self.output = q.get()
        if not name.empty():
            self.textNum = 1
            self.text = name.get()
        self.Disp()

    # 显示所有数据
    def Disp(self):
        if self.textNum:
            self.detect_Name.setText(self.text)
        cxmode = self.output[4]
        # 正常检测中
        if cxmode == 1:
            self.emo.setText(str(self.output[1]))  # 情绪
            self.bpm.setText(str(self.output[2]))  # 心率
            self.br.setText(str(self.output[6]))  # br
            self.SpO2.setText(str(self.output[7]))  # spo2
            # self.score.setText('--')  # 得分
            self.score.setText(str(self.output[3]))  # 得分
            self.djs = int(TOTALTIME - (self.output[5]) / FPS)
            # self.tishi.setText('正在检测，剩余时间为{}秒'.format(self.djs))
            self.xianshi.setText('正在检测')
            # self.progressBar.setValue(int(((TOTALTIME - self.djs) / TOTALTIME) * 100))
            self.progressBar.setValue(int(((TOTALTIME - self.djs) / 30) * 100))
            if self.djs % 31 != 0:
                self.timeControl = 0
            if self.djs % 31 == 0 and self.timeControl == 0:
                self.timeControl = 1
                self.save_history()

            img_list = img_lists.get()
            img = cv2.cvtColor(img_list, cv2.COLOR_BGR2RGB)
            qimg = array2qimage(img)
            pixmap = QPixmap(qimg)
            scaredPixmap = pixmap.scaled(450, 300, aspectRatioMode=Qt.KeepAspectRatio)
            self.frame.setPixmap(scaredPixmap)
            self.frame.show()

            wave = array2qimage(self.output[8])
            wavemap = QPixmap(wave)
            wavePixmap = wavemap.scaled(772, 87, aspectRatioMode=Qt.KeepAspectRatio)
            self.waveform.setPixmap(wavePixmap)
            self.waveform.show()

        # 检测完成
        elif cxmode == 2:
            self.emo.setText(str(self.output[1]))  # 情绪
            self.bpm.setText(str(self.output[2]))  # 心率
            self.br.setText(str(self.output[6]))  # br
            self.SpO2.setText(str(self.output[7]))  # spo2
            self.score.setText(str(self.output[3]))  # 得分
            if self.output[3] < 60:
                scoretext = '您的状态良好，请继续保持'
            elif self.output[3] < 80:
                scoretext = '您的状态一般，请多休息'
            elif self.output[3] < 101:
                scoretext = '您的冲动性较强，请注意'
            # self.tishi.setText('检测完成！按下结束按钮退出')
            self.xianshi.setText(scoretext)

            img_list = img_lists.get()
            img = cv2.cvtColor(img_list, cv2.COLOR_BGR2RGB)
            qimg = array2qimage(img)
            pixmap = QPixmap(qimg)
            scaredPixmap = pixmap.scaled(450, 300, aspectRatioMode=Qt.KeepAspectRatio)
            self.frame.setPixmap(scaredPixmap)
            # self.frame.setPixmap(QPixmap(qimg))
            self.frame.show()

        # 检测异常
        elif cxmode == 3:
            self.emo.setText('--')  # 情绪
            self.bpm.setText('--')  # 心率
            self.br.setText('--')  # br
            self.SpO2.setText('--')  # spo2
            self.score.setText('--')  # 得分
            self.djs = int(TOTALTIME - (self.output[5]) / FPS)
            # self.tishi.setText('正在检测，剩余时间为{}秒'.format(self.djs))
            self.xianshi.setText('未检测到人脸！请保持正对摄像头')

            img_list = img_lists.get()
            img = cv2.cvtColor(img_list, cv2.COLOR_BGR2RGB)
            qimg = array2qimage(img)
            pixmap = QPixmap(qimg)
            scaredPixmap = pixmap.scaled(450, 300, aspectRatioMode=Qt.KeepAspectRatio)
            self.frame.setPixmap(scaredPixmap)
            # self.frame.setPixmap(QPixmap(qimg))
            self.frame.show()

    # 开始
    def StartCamera(self):
        self.jieshu.setEnabled(True)
        self.jieshu.setChecked(False)
        self.kaishi.setEnabled(False)
        self.output = [0, 0, 0, 0, 0]
        con.put(1)
        con1.put(1)
        self.Timer.start(1)
        self.detect_Name.setText('--')  # 检测人姓名
        self.emo.setText('--')  # 情绪
        self.bpm.setText('--')  # 心率
        self.br.setText('--')  # br
        self.SpO2.setText('--')  # spo2
        self.score.setText('--')  # 得分
        self.timeControl = 0

    # 结束
    def StopCamera(self):
        self.kaishi.setEnabled(True)
        self.kaishi.setChecked(False)
        self.jieshu.setEnabled(False)
        con.put(0)
        con1.put(0)

        # 保存用户数据
        if self.output[4] == 2:
            detect_name = self.detect_Name.text()
            path = 'dataset/' + detect_name + ".txt"
            file = open(path, "a")  # 中文路径需要转成utf-8
            time_str = strftime('%Y.%m.%d %H:%M:%S')
            file.write(time_str)
            temptext = ":" + "  "
            file.write(temptext)
            bpm = "心率" + str(self.output[2]) + " 次/分  "
            file.write(bpm)
            br = "呼吸" + str(self.output[6]) + " 次  "
            file.write(br)
            SpO2 = "血氧" + str(self.output[7]) + " %  "
            file.write(SpO2)
            emo = "情绪" + str(self.output[1]) + "  "
            file.write(emo)
            score = "心率变异性" + str(self.output[3]) + " grade\n"
            file.write(score)
            file.close()

        self.emo.setText('--')  # 情绪
        self.bpm.setText('--')  # 心率
        self.br.setText('--')  # br
        self.SpO2.setText('--')  # spo2
        self.score.setText('--')  # 得分
        # self.tishi.setText('按下开始按钮进行检测')
        self.xianshi.setText('检测时请面朝摄像头')
        pixmap = QPixmap('face.png')
        scaredPixmap = pixmap.scaled(370, 280, aspectRatioMode=Qt.KeepAspectRatio)
        self.frame.setPixmap(scaredPixmap)
        self.frame.show()

        self.waveform.clear()
        while not q.empty():
            q.get()

        self.Timer.stop()

    # 保存用户数据
    def save_history(self):
        if self.detect_Name.text() == "":
            return
        if self.score.text() == "0":
            return
        detect_name = self.detect_Name.text()
        path = 'dataset/' + detect_name + ".txt"
        file = open(path, "a")  # 中文路径需要转成utf-8
        time_str = strftime('%Y.%m.%d %H:%M:%S')
        file.write(time_str)
        temptext = ":" + "  "
        file.write(temptext)
        bpm = "心率" + str(self.bpm.text()) + " 次/分  "
        file.write(bpm)
        br = "呼吸" + str(self.br.text()) + " 次  "
        file.write(br)
        SpO2 = "血氧" + str(self.SpO2.text()) + " %  "
        file.write(SpO2)
        emo = "情绪" + str(self.emo.text()) + "  "
        file.write(emo)
        score = "心率变异性" + str(self.score.text()) + " grade\n"
        file.write(score)
        file.close()
        print("save comlpleted")

    def on_history_clicked(self):
        dataset_path = "dataset/" + self.detect_Name.text() + ".txt"
        file = open(dataset_path, 'a+')  # 以读写模式打开
        file.seek(0)
        histories = file.readlines()
        self.m = Disp_History(histories)
        self.m.show()


if __name__ == '__main__':
    import multiprocessing

    # 该方法作用是阻止子进程运行其后面的代码
    multiprocessing.freeze_support()

    # ResNet表情相关参数
    resnet = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    resnet.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    resnet.add_module("resnet_block2", resnet_block(64, 128, 2))
    resnet.add_module("resnet_block3", resnet_block(128, 256, 2))
    resnet.add_module("resnet_block4", resnet_block(256, 512, 2))
    resnet.add_module("global_avg_pool", GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    resnet.add_module("fc", nn.Sequential(ResNet(), nn.Linear(512, 7)))

    # 管道相关
    frames = Queue(5000)
    q = Queue()
    img_lists = Queue(5000)

    con = Queue()
    con.put(0)

    con1 = Queue()
    con1.put(0)
    name = Queue()

    # 进程设置
    frame_get = Process(target=Td, args=(frames, con, img_lists))
    frame_use = Process(target=Heartbeat, args=(frames, q, con1, name))

    frame_get.daemon = True
    frame_use.daemon = True

    frame_get.start()
    frame_use.start()

    # 界面相关
    app = QApplication(sys.argv)
    w = mwindow()
    w.show()
    sys.exit(app.exec_())
