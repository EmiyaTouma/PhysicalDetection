# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cdjm.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_jiemian(object):
    def setupUi(self, jiemian):
        jiemian.setObjectName("jiemian")
        jiemian.resize(1235, 792)
        jiemian.setStyleSheet("QWidget#jiemian\n"
"{\n"
"    background-color: #f8fbfc;\n"
"}")
        self.gridLayout_8 = QtWidgets.QGridLayout(jiemian)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(-1, 10, -1, -1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(60, 10, -1, 25)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.waveform = QtWidgets.QLabel(jiemian)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.waveform.sizePolicy().hasHeightForWidth())
        self.waveform.setSizePolicy(sizePolicy)
        self.waveform.setStyleSheet("QLabel#waveform\n"
"{\n"
"    background-color:white;\n"
"    border: 2px solid grey;\n"
"    border-radius:15px;\n"
"}")
        self.waveform.setText("")
        self.waveform.setObjectName("waveform")
        self.verticalLayout_2.addWidget(self.waveform)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setContentsMargins(0, 25, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setContentsMargins(40, -1, 50, -1)
        self.horizontalLayout_10.setSpacing(7)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.kaishi = QtWidgets.QPushButton(jiemian)
        self.kaishi.setAutoFillBackground(False)
        self.kaishi.setStyleSheet("QPushButton#kaishi{\n"
"    width: 124px;\n"
"    height: 30px;\n"
"    background-color: white;\n"
"    font-size:20px;\n"
"    font-family:\"楷体\";\n"
"    border-style:solid;\n"
"    border-width: 1px;\n"
"    border-bottom-color: black;\n"
"    border-right-color: black;\n"
"    border-radius:6px;\n"
"}\n"
"\n"
"QPushButton#kaishi:checked {\n"
"    background-color: #8acdfa;\n"
"    border-top-color: black;\n"
"    border-left-color: black;\n"
"    border-bottom-color: white;\n"
"    border-right-color: white;\n"
"}\n"
"\n"
"QPushButton:enabled \n"
"{\n"
"    background-color: white;\n"
"}\n"
"\n"
"QPushButton#kaishi:pressed {\n"
"    background-color: #8acdfa;\n"
"    border-top-color: black;\n"
"    border-left-color: black;\n"
"    border-bottom-color: white;\n"
"    border-right-color: white;\n"
"}\n"
"\n"
"\n"
"\n"
"")
        self.kaishi.setCheckable(True)
        self.kaishi.setChecked(False)
        self.kaishi.setObjectName("kaishi")
        self.horizontalLayout_10.addWidget(self.kaishi)
        self.jieshu = QtWidgets.QPushButton(jiemian)
        self.jieshu.setStyleSheet("QPushButton#jieshu{\n"
"    width: 124px;\n"
"    height: 30px;\n"
"    background-color: white;\n"
"    font-size:20px;\n"
"    font-family:\"楷体\";\n"
"    border-style:solid;\n"
"    border-width: 1px;\n"
"    border-bottom-color: black;\n"
"    border-right-color: black;\n"
"    border-radius:6px;\n"
"}\n"
"\n"
"QPushButton#jieshu:checked {\n"
"    background-color: #8acdfa;\n"
"    border-top-color: black;\n"
"    border-left-color: black;\n"
"    border-bottom-color: white;\n"
"    border-right-color: white;\n"
"}\n"
"\n"
"QPushButton#jieshu:pressed {\n"
"    background-color: #8acdfa;\n"
"    border-top-color: black;\n"
"    border-left-color: black;\n"
"    border-bottom-color: white;\n"
"    border-right-color: white;\n"
"}\n"
"\n"
"\n"
"\n"
"")
        self.jieshu.setCheckable(True)
        self.jieshu.setChecked(False)
        self.jieshu.setObjectName("jieshu")
        self.horizontalLayout_10.addWidget(self.jieshu)
        self.verticalLayout_3.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label_4 = QtWidgets.QLabel(jiemian)
        self.label_4.setStyleSheet("QLabel{\n"
"    font-family:\"楷体\";\n"
"    font-size:15px\n"
"}")
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_11.addWidget(self.label_4)
        self.verticalLayout_3.addLayout(self.horizontalLayout_11)
        self.verticalLayout_3.setStretch(0, 5)
        self.verticalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.horizontalLayout_3.setStretch(0, 2)
        self.horizontalLayout_3.setStretch(1, 1)
        self.gridLayout_8.addLayout(self.horizontalLayout_3, 3, 0, 1, 2)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_24 = QtWidgets.QLabel(jiemian)
        self.label_24.setText("")
        self.label_24.setObjectName("label_24")
        self.horizontalLayout.addWidget(self.label_24)
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setContentsMargins(-1, 5, -1, 5)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.label = QtWidgets.QLabel(jiemian)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setStyleSheet("image: url(package/北理工.png);")
        self.label.setText("")
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.verticalLayout_11.addWidget(self.label)
        self.horizontalLayout.addLayout(self.verticalLayout_11)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setSpacing(7)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_3 = QtWidgets.QLabel(jiemian)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("QLabel\n"
"{\n"
"    font-family:\"隶书\";\n"
"    font-size:40px;\n"
"    font-weight: bold;\n"
"    qproperty-alignment: \'AlignVCenter | AlignHCenter\';\n"
"}")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_5.addWidget(self.label_3)
        self.horizontalLayout.addLayout(self.verticalLayout_5)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.detect_Name = QtWidgets.QLabel(jiemian)
        self.detect_Name.setStyleSheet("QLabel{\n"
"    font-size:1px;\n"
"    color:#f8fbfc;\n"
"}")
        self.detect_Name.setText("")
        self.detect_Name.setAlignment(QtCore.Qt.AlignCenter)
        self.detect_Name.setObjectName("detect_Name")
        self.horizontalLayout_2.addWidget(self.detect_Name)
        self.horizontalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 2)
        self.horizontalLayout.setStretch(2, 5)
        self.horizontalLayout.setStretch(3, 2)
        self.horizontalLayout.setStretch(4, 1)
        self.horizontalLayout_5.addLayout(self.horizontalLayout)
        self.gridLayout_8.addLayout(self.horizontalLayout_5, 0, 0, 1, 2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, 10, -1, -1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setContentsMargins(150, -1, 0, -1)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.frame = QtWidgets.QLabel(jiemian)
        self.frame.setStyleSheet("QLabel#frame\n"
"{\n"
"    background-color:rgb(230, 244, 255);\n"
"    border-radius:15px;\n"
"}")
        self.frame.setText("")
        self.frame.setAlignment(QtCore.Qt.AlignCenter)
        self.frame.setObjectName("frame")
        self.horizontalLayout_14.addWidget(self.frame)
        self.verticalLayout.addLayout(self.horizontalLayout_14)
        self.xianshi = QtWidgets.QLabel(jiemian)
        self.xianshi.setStyleSheet("QLabel#label_2{\n"
"    font-family:\"楷体\";\n"
"    font-size:20px\n"
"}")
        self.xianshi.setText("")
        self.xianshi.setAlignment(QtCore.Qt.AlignCenter)
        self.xianshi.setObjectName("xianshi")
        self.verticalLayout.addWidget(self.xianshi)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setContentsMargins(180, -1, 5, -1)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.progressBar = QtWidgets.QProgressBar(jiemian)
        self.progressBar.setStyleSheet("QProgressBar {\n"
"    text-align:right;\n"
"    border: 2px solid grey;\n"
"    border-radius: 5px;\n"
"    background-color: white;\n"
"    margin-right: 35px;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"    text-align:right;\n"
"    background-color: #51c2fc;\n"
"    \n"
"}")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setTextDirection(QtWidgets.QProgressBar.TopToBottom)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_12.addWidget(self.progressBar)
        self.verticalLayout.addLayout(self.horizontalLayout_12)
        self.verticalLayout.setStretch(0, 20)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.gridLayout_8.addLayout(self.verticalLayout, 1, 0, 2, 1)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setContentsMargins(100, -1, 200, 0)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.parameter_1 = QtWidgets.QFrame(jiemian)
        self.parameter_1.setStyleSheet("QFrame#parameter_1\n"
"{\n"
"    background-color:white;\n"
"    border-radius:20px;\n"
"}")
        self.parameter_1.setObjectName("parameter_1")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.parameter_1)
        self.horizontalLayout_17.setContentsMargins(11, -1, -1, -1)
        self.horizontalLayout_17.setSpacing(0)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.label_11 = QtWidgets.QLabel(self.parameter_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setStyleSheet("image: url(package/心率.png);")
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_17.addWidget(self.label_11)
        self.bpm = QtWidgets.QLabel(self.parameter_1)
        font = QtGui.QFont()
        font.setFamily("Tahoma")
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.bpm.setFont(font)
        self.bpm.setStyleSheet("QLabel\n"
"{\n"
"    font-family:Tahoma;\n"
"    font-size:30px;\n"
"    font-weight: bold;\n"
"    qproperty-alignment: \'AlignVCenter | AlignHCenter\';\n"
"}")
        self.bpm.setObjectName("bpm")
        self.horizontalLayout_17.addWidget(self.bpm)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setContentsMargins(-1, 8, -1, -1)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_21 = QtWidgets.QLabel(self.parameter_1)
        self.label_21.setStyleSheet("QLabel\n"
"{\n"
"    font-family:Tahoma;\n"
"    font-size:16px;\n"
"    font-weight: bold;\n"
"    color:#cacaca;\n"
"    qproperty-alignment: \'AlignVCenter | AlignHCenter\';\n"
"}")
        self.label_21.setObjectName("label_21")
        self.verticalLayout_10.addWidget(self.label_21)
        self.horizontalLayout_17.addLayout(self.verticalLayout_10)
        self.horizontalLayout_17.setStretch(0, 2)
        self.horizontalLayout_17.setStretch(1, 2)
        self.horizontalLayout_17.setStretch(2, 1)
        self.verticalLayout_12.addWidget(self.parameter_1)
        self.parameter_2 = QtWidgets.QFrame(jiemian)
        self.parameter_2.setStyleSheet("QFrame#parameter_2\n"
"{\n"
"    background-color:white;\n"
"    border-radius:20px;\n"
"}")
        self.parameter_2.setObjectName("parameter_2")
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout(self.parameter_2)
        self.horizontalLayout_19.setSpacing(0)
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.label_22 = QtWidgets.QLabel(self.parameter_2)
        self.label_22.setStyleSheet("image: url(package/S_健康呼吸首页.png);")
        self.label_22.setText("")
        self.label_22.setObjectName("label_22")
        self.horizontalLayout_19.addWidget(self.label_22)
        self.br = QtWidgets.QLabel(self.parameter_2)
        self.br.setStyleSheet("QLabel\n"
"{\n"
"    font-family:Tahoma;\n"
"    font-size:30px;\n"
"    font-weight: bold;\n"
"    qproperty-alignment: \'AlignVCenter | AlignHCenter\';\n"
"}")
        self.br.setObjectName("br")
        self.horizontalLayout_19.addWidget(self.br)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setContentsMargins(-1, 8, -1, -1)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_2 = QtWidgets.QLabel(self.parameter_2)
        self.label_2.setStyleSheet("QLabel\n"
"{\n"
"    font-family:Tahoma;\n"
"    font-size:16px;\n"
"    font-weight: bold;\n"
"    color:#cacaca;\n"
"    qproperty-alignment: \'AlignVCenter | AlignHCenter\';\n"
"}")
        self.label_2.setObjectName("label_2")
        self.verticalLayout_7.addWidget(self.label_2)
        self.horizontalLayout_19.addLayout(self.verticalLayout_7)
        self.horizontalLayout_19.setStretch(0, 2)
        self.horizontalLayout_19.setStretch(1, 2)
        self.horizontalLayout_19.setStretch(2, 1)
        self.verticalLayout_12.addWidget(self.parameter_2)
        self.parameter_3 = QtWidgets.QFrame(jiemian)
        self.parameter_3.setStyleSheet("QFrame#parameter_3\n"
"{\n"
"    background-color:white;\n"
"    border-radius:20px;\n"
"}")
        self.parameter_3.setObjectName("parameter_3")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout(self.parameter_3)
        self.horizontalLayout_18.setSpacing(0)
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.label_25 = QtWidgets.QLabel(self.parameter_3)
        self.label_25.setStyleSheet("image: url(package/血氧.png);")
        self.label_25.setText("")
        self.label_25.setObjectName("label_25")
        self.horizontalLayout_18.addWidget(self.label_25)
        self.SpO2 = QtWidgets.QLabel(self.parameter_3)
        self.SpO2.setStyleSheet("QLabel\n"
"{\n"
"    font-family:Tahoma;\n"
"    font-size:30px;\n"
"    font-weight: bold;\n"
"    qproperty-alignment: \'AlignVCenter | AlignHCenter\';\n"
"}")
        self.SpO2.setObjectName("SpO2")
        self.horizontalLayout_18.addWidget(self.SpO2)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setContentsMargins(-1, 8, -1, -1)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_5 = QtWidgets.QLabel(self.parameter_3)
        self.label_5.setStyleSheet("QLabel\n"
"{\n"
"    font-family:Tahoma;\n"
"    font-size:16px;\n"
"    font-weight: bold;\n"
"    color:#cacaca;\n"
"    qproperty-alignment: \'AlignVCenter | AlignHCenter\';\n"
"}")
        self.label_5.setObjectName("label_5")
        self.verticalLayout_8.addWidget(self.label_5)
        self.horizontalLayout_18.addLayout(self.verticalLayout_8)
        self.horizontalLayout_18.setStretch(0, 2)
        self.horizontalLayout_18.setStretch(1, 2)
        self.horizontalLayout_18.setStretch(2, 1)
        self.verticalLayout_12.addWidget(self.parameter_3)
        self.parameter_4 = QtWidgets.QFrame(jiemian)
        self.parameter_4.setStyleSheet("QFrame#parameter_4\n"
"{\n"
"    background-color:white;\n"
"    border-radius:20px;\n"
"}")
        self.parameter_4.setObjectName("parameter_4")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.parameter_4)
        self.horizontalLayout_16.setSpacing(0)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.label_27 = QtWidgets.QLabel(self.parameter_4)
        self.label_27.setStyleSheet("image: url(package/情绪.png);")
        self.label_27.setText("")
        self.label_27.setObjectName("label_27")
        self.horizontalLayout_16.addWidget(self.label_27)
        self.emo = QtWidgets.QLabel(self.parameter_4)
        self.emo.setStyleSheet("QLabel\n"
"{\n"
"    font-family:Tahoma;\n"
"    font-size:22px;\n"
"    font-weight: bold;\n"
"    qproperty-alignment: \'AlignVCenter | AlignHCenter\';\n"
"}")
        self.emo.setObjectName("emo")
        self.horizontalLayout_16.addWidget(self.emo)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setContentsMargins(-1, 8, -1, -1)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_6 = QtWidgets.QLabel(self.parameter_4)
        self.label_6.setStyleSheet("QLabel\n"
"{\n"
"    font-family:Tahoma;\n"
"    font-size:16px;\n"
"    font-weight: bold;\n"
"    color:#cacaca;\n"
"    qproperty-alignment: \'AlignVCenter | AlignHCenter\';\n"
"}")
        self.label_6.setObjectName("label_6")
        self.verticalLayout_9.addWidget(self.label_6)
        self.horizontalLayout_16.addLayout(self.verticalLayout_9)
        self.horizontalLayout_16.setStretch(0, 2)
        self.horizontalLayout_16.setStretch(1, 2)
        self.horizontalLayout_16.setStretch(2, 1)
        self.verticalLayout_12.addWidget(self.parameter_4)
        self.parameter_5 = QtWidgets.QFrame(jiemian)
        self.parameter_5.setStyleSheet("QFrame#parameter_5\n"
"{\n"
"    background-color:white;\n"
"    border-radius:20px;\n"
"}")
        self.parameter_5.setObjectName("parameter_5")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.parameter_5)
        self.horizontalLayout_13.setSpacing(0)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.label_29 = QtWidgets.QLabel(self.parameter_5)
        self.label_29.setStyleSheet("image: url(package/waveform_path_ecg.png);")
        self.label_29.setText("")
        self.label_29.setObjectName("label_29")
        self.horizontalLayout_13.addWidget(self.label_29)
        self.score = QtWidgets.QLabel(self.parameter_5)
        self.score.setStyleSheet("QLabel\n"
"{\n"
"    font-family:Tahoma;\n"
"    font-size:30px;\n"
"    font-weight: bold;\n"
"    qproperty-alignment: \'AlignVCenter | AlignHCenter\';\n"
"}")
        self.score.setObjectName("score")
        self.horizontalLayout_13.addWidget(self.score)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setContentsMargins(-1, 8, -1, -1)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_7 = QtWidgets.QLabel(self.parameter_5)
        self.label_7.setStyleSheet("QLabel\n"
"{\n"
"    font-family:Tahoma;\n"
"    font-size:16px;\n"
"    font-weight: bold;\n"
"    color:#cacaca;\n"
"    qproperty-alignment: \'AlignVCenter | AlignHCenter\';\n"
"}")
        self.label_7.setObjectName("label_7")
        self.verticalLayout_6.addWidget(self.label_7)
        self.horizontalLayout_13.addLayout(self.verticalLayout_6)
        self.horizontalLayout_13.setStretch(0, 2)
        self.horizontalLayout_13.setStretch(1, 2)
        self.horizontalLayout_13.setStretch(2, 1)
        self.verticalLayout_12.addWidget(self.parameter_5)
        self.gridLayout_8.addLayout(self.verticalLayout_12, 1, 1, 2, 1)
        self.gridLayout_8.setColumnStretch(0, 1)
        self.gridLayout_8.setColumnStretch(1, 1)
        self.gridLayout_8.setRowStretch(0, 100)
        self.gridLayout_8.setRowStretch(1, 360)
        self.gridLayout_8.setRowStretch(2, 100)
        self.gridLayout_8.setRowStretch(3, 100)

        self.retranslateUi(jiemian)
        QtCore.QMetaObject.connectSlotsByName(jiemian)

    def retranslateUi(self, jiemian):
        _translate = QtCore.QCoreApplication.translate
        jiemian.setWindowTitle(_translate("jiemian", "Form"))
        self.kaishi.setText(_translate("jiemian", "开始"))
        self.jieshu.setText(_translate("jiemian", "结束"))
        self.label_4.setText(_translate("jiemian", "BIT"))
        self.label_3.setText(_translate("jiemian", "成像非接触式生理参数监测系统"))
        self.bpm.setText(_translate("jiemian", "70"))
        self.label_21.setText(_translate("jiemian", "HR"))
        self.br.setText(_translate("jiemian", "16"))
        self.label_2.setText(_translate("jiemian", "RR"))
        self.SpO2.setText(_translate("jiemian", "95"))
        self.label_5.setText(_translate("jiemian", "SpO2"))
        self.emo.setText(_translate("jiemian", "angry"))
        self.label_6.setText(_translate("jiemian", "EMO"))
        self.score.setText(_translate("jiemian", "47.5"))
        self.label_7.setText(_translate("jiemian", "HRV"))
import jiemiantubiao_rc
