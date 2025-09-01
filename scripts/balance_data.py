import os
import shutil
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
import numpy as np
from PIL import Image

def collect_data(data_dir):
    X, y, paths = [], [], []
    for emotion in os.listdir(data_dir):
        emotion_path = os.path.join(data_dir, emotion)
        for file in os.listdir(emotion_path):
            fpath = os.path.join(emotion_path, file)
            X.append(np.array(Image.open(fpath).resize((48, 48))).flatten())
            y.append(emotion)
            paths.append(fpath)
    return np.array(X), np.array(y), paths

def balance_dataset(train_path, save_path):
    X, y, paths = collect_data(train_path)
    ros = RandomOverSampler()
    X_res, y_res = ros.fit_resample(X, y)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    for emotion in np.unique(y_res):
        os.makedirs(os.path.join(save_path, emotion))

    count = {e: 0 for e in np.unique(y_res)}
    for idx, label in enumerate(y_res):
        fname = f"{label}_{count[label]}.jpg"
        out_path = os.path.join(save_path, label, fname)
        image = Image.fromarray(X_res[idx].reshape(48, 48).astype('uint8'))
        image.save(out_path)
        count[label] += 1

    print("Balanced data saved to:", save_path)

balance_dataset('fer2013/train', 'balanced_data/train')