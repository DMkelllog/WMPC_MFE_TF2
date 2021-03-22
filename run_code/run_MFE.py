# Wafer map pattern classification using MFE

import pickle
import os

import pickle
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

BATCH_SIZE = 32
MAX_EPOCH = 1000
TRAIN_SIZE_LIST = [500, 5000, 50000, 162946]
LEARNING_RATE = 1e-4

early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

with open('../data/X_MFE.pickle', 'rb') as f:
    X_mfe = pickle.load(f)
with open('../data/y.pickle', 'rb') as f:
    y = pickle.load(f)
    
def FNN(lr=1e-4):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='tanh'),
        tf.keras.layers.Dense(128, activation='tanh'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])
    model.compile(optimizer= tf.keras.optimizers.Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

REP_ID = 0
RAN_NUM = 27407+REP_ID

for TRAIN_SIZE_ID in range(4):
    TRAIN_SIZE = TRAIN_SIZE_LIST[TRAIN_SIZE_ID]

    X_trnval, X_tst, y_trnval, y_tst =  train_test_split(X_mfe, y, 
                                                         test_size=10000, random_state=RAN_NUM)

    # Randomly sample train set for evaluation at various training set size
    if TRAIN_SIZE == X_trnval.shape[0]:
        pass
    else:
        X_trnval,_ , y_trnval, _ = train_test_split(X_trnval, y_trnval, 
                                                    train_size=TRAIN_SIZE, random_state=RAN_NUM)
    # Get unique labels in training set. Some labels might not appear in small training set.
    labels = np.unique(y_trnval)

    scaler = StandardScaler()
    X_trnval_scaled = scaler.fit_transform(X_trnval)
    X_tst_scaled = scaler.transform(X_tst)


    model = FNN(lr=LEARNING_RATE)

    y_trnval = tf.keras.utils.to_categorical(y_trnval)
    y_tst = tf.keras.utils.to_categorical(y_tst)
    log = model.fit(X_trnval_scaled, y_trnval, validation_split=0.2, batch_size=BATCH_SIZE,
              epochs=MAX_EPOCH, callbacks=[early_stopping], verbose=0)
    y_trnval_hat= model.predict(X_trnval_scaled)
    y_tst_hat= model.predict(X_tst_scaled)

    macro = f1_score(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1), labels=labels, average='macro')
    micro = f1_score(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1), labels=labels, average='micro')
    cm = confusion_matrix(np.argmax(y_tst, 1), np.argmax(y_tst_hat, 1))

    filename = '../result/WMPC_'+'MFE_'+str(TRAIN_SIZE)+'_'+str(REP_ID)+'_'

    with open(filename+'softmax.pickle', 'wb') as f:
        pickle.dump([y_trnval_hat, y_tst_hat], f)
    with open(filename+'f1_score.pickle', 'wb') as f:
        pickle.dump([macro, micro, cm], f)

    print('train size:', TRAIN_SIZE,
          'rep_id:', REP_ID,
          'macro:', np.round(macro, 4), 
          'micro:', np.round(micro, 4))