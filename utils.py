import os
import glob
import csv
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import device, io, float32
from tensorflow.image import decode_jpeg, decode_png, convert_image_dtype, resize
from tensorflow.keras import models, backend



def move_data(path_to_save, folder, data):

    for d in data:

        path_to_folder = os.path.join(path_to_save, folder)

        if not os.path.isdir(path_to_folder):
            os.makedirs(path_to_folder)
        
        shutil.copy(d, path_to_folder)


def split_data(path_to_data, path_to_train, path_to_val, path_to_test):
    
    folders = os.listdir(path_to_data)

    for folder in folders:
        
        full_path = os.path.join(path_to_data, folder)
        image_paths = glob.glob(os.path.join(full_path, '*.jpg'))

        X_train, X = train_test_split(image_paths, test_size=0.3, random_state=11)
        X_val, X_test = train_test_split(X, test_size=0.5, random_state=11)

        move_data(path_to_train, folder, X_train)
        move_data(path_to_val, folder, X_val)
        move_data(path_to_test, folder, X_test)

def data_generators(path_to_train, path_to_val, path_to_test, target_size, batch_size=8):

    train_preprocessor = ImageDataGenerator(
        rescale=1/255,
        rotation_range=10, 
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    test_preprocessor = ImageDataGenerator(
        rescale = 1 / 255
    )

    train_generator = train_preprocessor.flow_from_directory(
        path_to_train, 
        class_mode='categorical',
        target_size=target_size,
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size,
        interpolation='bilinear'
    )

    val_generator = test_preprocessor.flow_from_directory(
        path_to_val, 
        class_mode='categorical',
        target_size=target_size,
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size,
        interpolation='bilinear'
    )

    test_generator = test_preprocessor.flow_from_directory(
        path_to_test,
        class_mode='categorical',
        target_size=target_size,
        color_mode='rgb',
        shuffle=False,
        batch_size=batch_size,
        interpolation='bilinear'
    )

    return train_generator, val_generator, test_generator

def touch(path='', filename=''):
    file_path = os.path.join(path, filename)
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(file_path, "a")

def write_accuracy_log(path, val_acc, test_acc, message=''):

    f = open(path, 'a', newline='')
    writer = csv.writer(f)
    writer.writerow((message))
    writer.writerow((val_acc))
    writer.writerow((test_acc))
    f.close()

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10)
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def model_evaluation_cmf(model, test_path, log_path, confmat_path, nbr_classes, batch, initializer, name):
    
    y_true = list()
    y_true_index = list()
    y_pred = list()

    lr = str(model.optimizer.get_config()['learning_rate'])
    optimizer = model.optimizer.get_config()['name']

    print(lr)

    folders = os.listdir(test_path)

    for i in range(len(folders)):
        y_true_sub = list()
        for j in range(len(folders)):
            if j == i:
                y_true_sub.append(1)
            else:
                y_true_sub.append(0)

        full_path = os.path.join(test_path, folders[i])
        image_paths = glob.glob(os.path.join(full_path, '*.jpg'))

        for image in image_paths:
            y_pred_sub = list()

            imgExtension = 'jpg'
            image = io.read_file(image)
    
            if imgExtension in ['jpeg', 'jpg']:
                image = decode_jpeg(image, channels=3)
            elif imgExtension == 'png':
                image = decode_png(image, channels=3)

            image = convert_image_dtype(image, dtype=float32)
            image = resize(image, [299, 299])
            image = np.expand_dims(image, axis=0)

            with device('/GPU:0'):
                predictions = model.predict(image)
                predictions = np.argmax(predictions[0])
            
            for j in range(len(folders)):
                if j == predictions:
                    y_pred_sub.append(1)
                else:
                    y_pred_sub.append(0)
            
            y_true.append(y_true_sub)
            y_true_index.append(i)
            y_pred.append(y_pred_sub)

    
    y_pred_index = list()
    
    for l in y_pred:
        y_pred_index.append(np.argmax(l))

    cm = confusion_matrix(y_true_index, y_pred_index, labels=[i for i in range(nbr_classes)])

    spesificity = 0

    for i in range(nbr_classes):

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for j in range(len(y_pred_index)):
            if y_true_index[j] == i and y_true_index[j] == y_pred_index[j]:
                tp += 1
            elif y_true_index[j] != i and y_true_index[j] != y_pred_index[j]:
                tn += 1
            elif y_true_index[j] == i and y_true_index[j] != y_pred_index[j]:
                fn += 1
            else:
                fp += 1

        spesificity += tn / (tn + fp)
    
    spesificity /= nbr_classes

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    if name == 'final':
        f = open(log_path, 'a', newline='')
        writer = csv.writer(f)
        writer.writerow(([optimizer,batch,initializer[0],initializer[1],lr,accuracy,precision,recall,spesificity,f1]))
        f.close()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(nbr_classes)])
    disp.plot()
    disp.figure_.savefig(f'log/{optimizer}/{batch}/{initializer[0]}/{initializer[1]}/cm-{name}-{optimizer}-{batch}-{lr}.jpg')

            
if __name__ == '__main__':

    if True:
        path_to_data = os.getcwd() + '/data'
        path_to_train = os.getcwd() + '/data/train'
        path_to_val = os.getcwd() + '/data/val'
        path_to_test = os.getcwd() + '/data/test'

        split_data(path_to_data, path_to_train, path_to_val, path_to_test)
    
    if False:
        path_to_train = os.getcwd() + '/data/train'
        path_to_val = os.getcwd() + '/data/val'
        path_to_test = os.getcwd() + '/data/test'

        train_generator, val_generator, test_generator = data_generators(path_to_train, path_to_val, path_to_test, target_size=(299, 299))
        
        # imgs, labels = next(train_generator)

        # plotImages(imgs)


    # touch('test', 'test.csv')
    
