from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from keras.regularizers import l2
from keras.backend import sigmoid
import tensorflow as tf
import os
import numpy as np
import sys
import argparse
import time
import math
from keras.utils import get_custom_objects
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from sklearn.model_selection import StratifiedKFold
# from keras_multi_head import MultiHeadAttention
from sklearn import metrics
from keras import regularizers
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.layers import Layer
from keras import initializers
from feature.AAindex.AAindex_sl import AAindex_hm_encoding, AAindex_ms_encoding, col_delete
from feature.word2vec.w2v_fea import w2v_fea_hm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf

def normalization_layer(input_layer):
#    output_layer = Lambda(lambda x: x - K.mean(x))(input_layer)均值归零
    output_layer = Lambda(lambda x: x / K.max(x))(input_layer)
    return output_layer

# 定义激活函数
def swish(x, beta=1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': swish})


def Twoclassfy_evalu(y_test, y_predict):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []
    for i in range(len(y_test)):
        if y_predict[i] > 0.5 and y_test[i] == 1:
            TP += 1
        if y_predict[i] > 0.5 and y_test[i] == 0:
            FP += 1
            FP_index.append(i)
        if y_predict[i] < 0.5 and y_test[i] == 1:
            FN += 1
            FN_index.append(i)
        if y_predict[i] < 0.5 and y_test[i] == 0:
            TN += 1
    Sn = TP / (TP + FN)
    Sp = TN / (FP + TN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TN + FN) * (FP + TN) * (TP + FN) * (TP + FP))
    Acc = (TP + TN) / (TP + FP + TN + FN)
    fpr,tpr,thresholds = metrics.roc_curve(y_test,y_predict,pos_label=1)  #poslabel正样本的标签
    auc = metrics.auc(fpr,tpr)

    return Sn, Sp, Acc, MCC, auc


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def step_decay(epoch):
    initial_lrate = 0.0005
    drop = 0.8
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

## 1
def createModel1():
        word_input = Input(shape=(21, 640), name='word_input')
        # path1
        overallResult = Convolution1D(filters=128, kernel_size=3, padding='same', activation="relu")(word_input)
        overallResult = MaxPooling1D(pool_size=2)(overallResult)
        flatten = Flatten()(overallResult)
        # Path 2
        overallResult1 = Convolution1D(filters=128, kernel_size=5, padding='same', activation='relu')(word_input)
        overallResult1 = MaxPooling1D(pool_size=2)(overallResult1)
        overallResult1 = GlobalMaxPooling1D()(overallResult1)

        merged = Concatenate()([normalization_layer(overallResult1), normalization_layer(flatten)])

        dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(merged)
        dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dense1)
        ss_output = Dense(1, activation='sigmoid', name='ss_output')(dense2)
        #
        return Model(inputs=[word_input], outputs=[ss_output])

def createModel2():
     word_input = Input(shape=(21, 29), name='word_input')

     x = TimeDistributed(Dense(units=32, activation="tanh"))(word_input)
     x = TimeDistributed(Dense(units=16, activation="tanh"))(x)

     overallResult = Convolution1D(filters=32, kernel_size=1, padding='same', activation="relu")(x)
     overallResult = MaxPooling1D(pool_size=2)(overallResult)
     flatten = Flatten()(overallResult)

     overallResult1 = Bidirectional(LSTM(units=32, return_sequences=True, activation='tanh'))(x)
     overallResult1 = Bidirectional(LSTM(units=16, return_sequences=True, activation='tanh'))(overallResult1)
     flatten1 = Flatten()(overallResult1)

     merged = Concatenate()([normalization_layer(flatten), normalization_layer(flatten1)])

     overallResult = Dense(32, activation='sigmoid')(merged)
     ss_output = Dense(1, activation='sigmoid', name='ss_output')(overallResult)

     return Model(inputs=[word_input], outputs=[ss_output])


def createModel3():
    word_input = Input(shape=(20, 40), name='word_input')

    x = TimeDistributed(Dense(units=32, activation="tanh"))(word_input)
    x = TimeDistributed(Dense(units=16, activation="tanh"))(x)

    overallResult = Convolution1D(filters=32, kernel_size=1, padding='same', activation="relu")(x)
    overallResult = MaxPooling1D(pool_size=2)(overallResult)
    flatten = Flatten()(overallResult)

    overallResult1 = Bidirectional(LSTM(units=32, return_sequences=True, activation='tanh'))(x)
    overallResult1 = Bidirectional(LSTM(units=16, return_sequences=True, activation='tanh'))(overallResult1)
    flatten1 = Flatten()(overallResult1)

    merged = Concatenate()([normalization_layer(flatten), normalization_layer(flatten1)])

    overallResult = Dense(32, activation='sigmoid')(merged)
    ss_output = Dense(1, activation='sigmoid', name='ss_output')(overallResult)

    return Model(inputs=[word_input], outputs=[ss_output])

## ============================================= 数据 ========================================
file = r'D:\pycharm\project\OGlc\tset\hfz\Ind_M1.fasta'

## AAindex
encodings = AAindex_hm_encoding(file)
X= np.array(encodings).reshape(-1, 609)#29*21
X1 = X.reshape(X.shape[0], 21, 29)
X1 = np.array(X1, dtype=float)
print(X1.shape)

##  w2v
w2v_mod = r'D:\pycharm\project\OGlc\feature\word2vec\w2v_ms_w3_v40.model'
X2 = w2v_fea_hm(file, w2v_mod).reshape(X.shape[0], 20, 40)
print(X2.shape)

## ============================================= 模型 ========================================

M2 = createModel2()
M2.load_weights(r'D:\pycharm\project\OGlc\data(未去冗余)\HUMAN\savemodel\data2\model2.h5')
y_predict_test2 = M2.predict({'word_input': X1})

M3 = createModel3()
M3.load_weights(r'D:\pycharm\project\OGlc\data(未去冗余)\HUMAN\savemodel\data5\model3.h5')
y_predict_test3 = M3.predict({'word_input': X2})

## Esm2
# 加载保存的数据
X11 = np.load(r'D:\pycharm\project\OGlc\data(未去冗余)\aftercoding\h_data\HUMAN\esm\pos.npy')
X22 = np.load(r'D:\pycharm\project\OGlc\data(未去冗余)\aftercoding\h_data\HUMAN\esm\neg1.npy')

#合并 X1 和 X2，假设它们的维度匹配
X = np.concatenate((X11, X22), axis=0)
#创建对应的标签，假设 X1 对应标签为 1，X2 对应标签为 0
y = np.concatenate((np.ones(X11.shape[0]), np.zeros(X22.shape[0])))
#将数据集分割为训练数据集和测试数据集（按照 9:1 的比例）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train = X_train.reshape(-1, X_train.shape[-1])#第一、二维展平
X_test = X_test.reshape(-1, X_test.shape[-1])

pca = PCA(n_components=640)

word = pca.fit_transform(X_train)
#test_word = pca.transform(X_test)

word = word.reshape(4852, 21, 640)#H4852\M1713
train_label = y_train#训练集标签

#data3 = np.load(r'F:\Project\OGlc\data(未去冗余)\others\other\pos_all.npy')
#data4 = np.load(r'F:\Project\OGlc\data(未去冗余)\others\other\neg_all.npy')

#X3 = data3 #pos的esm
#X4 = data4 #neg的esm

#X0 = np.concatenate((X3, X4), axis=0)

X0 = np.load(r'D:\pycharm\project\OGlc\tset\hfz\Ind_M1.npy')
#print(X0.shape)
#y0 = np.concatenate((np.ones(X3.shape[0]), np.zeros(X4.shape[0])))
y0 = np.concatenate((np.ones(143), np.zeros(715)))
test_word = X0.reshape(-1, X0.shape[-1])#第一、二维展平
print(test_word.shape)
test_word = pca.transform(test_word)
print(test_word.shape)
test_word = test_word.reshape(858, 21, 640)#h3491\m12384\zhuyan6325\1918\1442\3489\400
test_label = y0#测试集标签

# 保存模型权重的路径
MODEL_PATH = r'D:\pycharm\project\OGlc\data(未去冗余)\HUMAN\savemodel\data1'
# 设置模型权重文件的路径

filename = 'model1.h5'
filepath = os.path.join(MODEL_PATH, filename)
seed = 7
np.random.seed(seed)
KF = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
SN = []
SP = []
ACC = []
MCC = []
Precision = []
F1_score = []
AUC = []

tprs = []
aucs1 = []
mean_fpr = np.linspace(0, 1, 100)
batchSize = 32
maxEpochs = 200

# data为数据集,利用KF.split划分训练集和测试集
for train_index, val_index in KF.split(word, train_label):
    # 建立模型，并对训练集进行测试，求出预测得分
    # 划分训练集和测试集
    x_train_word, x_val = word[train_index], word[val_index]
    y_train_word, y_val = train_label[train_index], train_label[val_index]

    model = createModel1()#更换模型
    model.count_params()
    model.summary()
    model.compile(optimizer='adam',
                  loss={'ss_output': 'binary_crossentropy'}, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True,
                                 mode='auto')

    history = model.fit(
        {'word_input': x_train_word},
        {'ss_output': y_train_word},
        epochs=maxEpochs,
        batch_size=batchSize,
        callbacks=[EarlyStopping(monitor='val_accuracy', patience=20, verbose=0, mode='auto'),
                   checkpoint, LearningRateScheduler(step_decay)],
        verbose=2,
        validation_data=({'word_input': x_val},
                         {'ss_output': y_val}),
        shuffle=True)

    model_save_path = os.path.join(MODEL_PATH, filename)
    # 保存模型
    model.save(model_save_path)

    score = model.evaluate({'word_input': x_val,}, y_val)

    y_pred1 = model.predict({'word_input': x_val})

    (Sn1, Sp1, Acc1, MCC1,auc) = Twoclassfy_evalu(y_val, y_pred1)

    SN.append(Sn1)
    SP.append(Sp1)
    MCC.append(MCC1)
    ACC.append(Acc1)
    AUC.append(auc)

print('SN', SN)
print('SP', SP)
print('ACC', ACC)
print('MCC', MCC)
print('AUC', AUC)

meanSN = np.mean(SN)
meanSP = np.mean(SP)
meanACC = np.mean(ACC)
meanMCC = np.mean(MCC)
meanAUC = np.mean(AUC)

print("meanSN", round(meanSN, 4))
print("meanSP", round(meanSP, 4))
print("meanACC", round(meanACC, 4))
print("meanMCC", round(meanMCC, 4))
print("meanAUC", round(meanAUC, 4))

M1 = createModel1()#更换模型
M1.load_weights(model_save_path)
y_predict_test1 = M1.predict({'word_input': test_word})

## ============================================= 集成 ========================================
sum_lst = []

for index, item in enumerate(y_predict_test1):

    sum_lst.append(((item*0.5) + (y_predict_test2[index])*0.3 + (y_predict_test3[index])*0.2))

b = np.array(sum_lst)
# 输出预测结果，>=0.5为O-糖基化位点，<0.5不是
print(b.shape)#1442的标签

(Sn, Sp, Acc, MCC, AUC) = Twoclassfy_evalu(test_label, b)

print(round(Sn, 4))
print(round(Sp, 4))
print(round(Acc, 4))
print(round(MCC, 4))
print(round(AUC, 4))

