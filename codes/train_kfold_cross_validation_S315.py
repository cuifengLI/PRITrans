import tensorflow as tf
import numpy as np
import gc
from model import get_model
import os
import pandas as pd
import sys
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import warnings
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import statistics

warnings.filterwarnings("ignore")


filename = 'F:\\protein_stability\\result\\XXXXXX.xlsx'

def op_toexcel(data, filename):
    if os.path.exists(filename):
        wb = op.load_workbook(filename)
        ws = wb.worksheets[0]
        ws.append(data)  
        wb.save(filename)

    else:
        wb = op.Workbook()  
        ws = wb['Sheet']  
        ws.append(['MSE', 'MAE', 'RMSE', 'R2', 'PCC', 'P_value', 'Delta'])
        ws.append(data)  
        wb.save(filename)

def data_generator(train_esm, train_prot, train_y, batch_size):
    L = train_esm.shape[0]

    while True:
        for i in range(0, L, batch_size):
            batch_esm = train_esm[i:i + batch_size].copy()
            batch_prot = train_prot[i:i + batch_size].copy()
            batch_y = train_y[i:i + batch_size].copy()

            yield ([batch_esm, batch_prot], batch_y)


def cross_validation(train_esm, train_prot, train_label, valid_esm, valid_prot, valid_label, test_esm, test_prot,
                     test_label, k, i):

    train_size = train_label.shape[0]
    val_size = valid_label.shape[0]
    batch_size = 16
    train_steps = train_size // batch_size
    val_steps = val_size // batch_size

    qa_model = get_model()
    valiBestModel = 'F:\\protein_stability\\save_model\\Kfold\\XXXXXX.h5'

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=valiBestModel,  
        monitor='val_loss',  
        save_weights_only=True,  
        verbose=1,  
        save_best_only=True,
        mode='min',
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,  
        verbose=0,  
        mode='min' 
    )

    train_generator = data_generator(train_esm, train_prot, train_label, batch_size)
    val_generator = data_generator(valid_esm, valid_prot, valid_label, batch_size)

    history_callback = qa_model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=60,
        verbose=1,
        callbacks=[checkpointer, early_stopping],
        validation_data=val_generator,
        validation_steps=val_steps,
        shuffle=True, 
        workers=1
    )
    
    rmse = history_callback.history['root_mean_squared_error']
    val_rmse = history_callback.history['val_root_mean_squared_error']
    epochs = range(1, len(rmse) + 1)

    plt.plot(epochs, rmse, 'bo', label='Training rmse')
    plt.plot(epochs, val_rmse, 'b', label='Validation rmse')
    plt.title(f'Test{i}Fold{k} Training And Validation rmse')
    plt.xlabel('60 Epochs 50 patience')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

    loss = history_callback.history['loss']
    val_loss = history_callback.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title(f'TEST{i}Fold{k} Training And Validation LOSS')
    plt.xlabel('60 Epochs 50 patience')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    train_generator.close()
    val_generator.close()
    del train_generator
    del val_generator
    gc.collect() 

    print(f"Fold {k} - Testing:")  
    
    test_pred = qa_model.predict([test_esm, test_prot]).reshape(-1, )
    evaluate_regression(test_pred, test_label)  

    y_true_flat = np.ravel(test_label)
    y_pred_flat = np.ravel(test_pred)
    result_df = pd.DataFrame({
        'TRUE_label': y_true_flat,
        'PRED_label': y_pred_flat
    })
    result_end = 'F:\\protein_stability\\result\\XXXXXX.csv'
    result_df.to_csv(result_end, index=False)


def evaluate_regression(test_pred, test_label):
    y_pred = test_pred
    y_true = test_label

    print("TRUE VALUES：" + str(test_label))
    print("PRED VALUES：" + str(test_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pearsonr_corr, p_value = pearsonr(y_true, y_pred)
    delta = np.mean(np.abs(y_true - y_pred)) 

    print(f"\nFold {k} - Mean Squared Error (MSE): {mse:.4f}")
    print(f"Fold {k} - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Fold {k} - RMSE: {rmse:.4f}")
    print(f"Fold {k} - R-squared (R2) Score: {r2:.4f}")
    print(f"Fold {k} - PCC: {pearsonr_corr:.4f}")
    print(f"Fold {k} - P-value: {p_value:.4f}")
    print(f"Fold {k} - Delta: {delta:.4f}")

    result = mse, mae, rmse, r2, pearsonr_corr, p_value, delta
    op_toexcel(result, filename)


if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    all_esm = np.lib.format.open_memmap('../features_npy/S394_forward_181_esm.npy')
    all_prot = np.lib.format.open_memmap('../features_npy/S394_forward_181_prot.npy')
    all_label = np.lib.format.open_memmap('../features_npy/S394_forward_181_label.npy')

   
    all_label = all_label.astype(np.float)
    print(all_label.dtype)

    # S79
    all_esm_train, all_esm_inde, all_prot_train, all_prot_inde, all_label_train, all_label_inde = train_test_split(
        all_esm, all_prot, all_label, test_size=0.2, random_state=42)
    print(all_label_train.shape)
    print(all_label)
    print(all_label_inde.shape)
    print(all_label_inde)

    for i in range(1):
        cv = KFold(n_splits=10, shuffle=True, random_state=42)  
        k = 1
        for train_index, test_index in cv.split(all_esm_train, all_label_train):
            train_ESM = all_esm_train[train_index]
            train_Prot = all_prot_train[train_index]
            train_Y = all_label_train[train_index]

            train_ESM, valid_ESM, train_Prot, valid_Prot, train_Y, valid_Y = train_test_split(train_ESM, train_Prot,
                                                                                              train_Y, test_size=0.1,
                                                                                              random_state=42)

            test_ESM = all_esm_train[test_index]
            test_Prot = all_prot_train[test_index]
            test_Y = all_label_train[test_index]

            cross_validation(train_ESM, train_Prot, train_Y, valid_ESM, valid_Prot, valid_Y, test_ESM, test_Prot,
                             test_Y, k, i)

            k += 1
