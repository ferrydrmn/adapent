import os
import csv
import pandas as pd
import tensorflow as tf
from tensorflow.nn import lrn
from tensorflow.keras import models, losses
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import HeUniform, HeNormal, GlorotUniform, GlorotNormal

from utils import data_generators, touch, write_accuracy_log, model_evaluation_cmf
from inception_resnet_v2 import inception_resnet_v2

if __name__ == '__main__':
    
    path_to_train = os.getcwd() + '/data/train'
    path_to_val = os.getcwd() + '/data/val'
    path_to_test = os.getcwd() + '/data/test'

    epochs = 100

    batch_size = [16, 32, 64]
    target_size = (299, 299)
    lrs = [0.01, 0.001, 0.0001, 0.00001]

    optimizers = ['adam', 'rmsprop']

    initializers_A = [['heuniform', HeUniform(seed=11)], ['henormal', HeNormal(seed=11)]]
    initializers_B = [['glorotuniform', GlorotUniform(seed=11)], ['glorotnormal', GlorotNormal(seed=11)]]

    if not os.path.isfile('log/best_model.csv'):

        touch('log', 'best_model.csv')
        f = open('log/best_model.csv', 'a', newline='')
        writer = csv.writer(f)
        writer.writerow((['optimizer','batch', 'initializer_A', 'initializer_B', 'lr', 'accuracy', 'precision', 'recall', 'spesificity', 'f1']))
        f.close()
    
    df = pd.read_csv('log/best_model.csv')

    for optimizer in optimizers:

        for batch in batch_size:
            
            train_generator, val_generator, test_generator = data_generators(path_to_train, 
                path_to_val, path_to_test, target_size, batch)
            
            nbr_classes = train_generator.num_classes 

            for initializer_A in initializers_A:

                for initializer_B in initializers_B:

                    path_to_save_weights = f'weights/{optimizer}/{batch}/{initializer_A[0]}/{initializer_B[0]}'

                    if not os.path.isdir(path_to_save_weights):
                        os.makedirs(path_to_save_weights)
                    
                    for lr in lrs:

                        if optimizer == 'adam':
                            opt = Adam(
                                learning_rate=lr,
                                beta_1=0.9,
                                beta_2=0.999,
                                epsilon=1e-08,
                            )
                        else:
                            opt = RMSprop(
                                learning_rate=lr,
                                rho=0.999,
                                epsilon=1e-08,
                            )

                        log_name = f'{lr}-log.csv'
                        log_path = f'log/{optimizer}/{batch}/{initializer_A[0]}/{initializer_B[0]}'
                        path_to_log_accuracy = log_path + '/' + log_name

                        if not os.path.exists(path_to_log_accuracy):
                            
                            touch(log_path, log_name)
                        
                        log_name = f'conf-{batch}.csv'
                        log_path = f'log/{optimizer}/{batch}'
                        path_to_log_conf = log_path + '/' + log_name
                            
                        if not os.path.exists(path_to_log_conf):

                            touch(log_path, log_name)
                            
                            f = open(log_path + '/' + log_name, 'a', newline='')
                            writer = csv.writer(f)
                            writer.writerow((['optimizer','batch', 'initializer_A', 'initializer_B', 'lr', 'accuracy', 'precision', 'recall', 'spesificity', 'f1']))
                            f.close()

                        weight_saver = ModelCheckpoint(
                            path_to_save_weights + f'/{lr}/best.h5', 
                            monitor='val_accuracy',
                            mode='max',
                            save_best_only=True,
                            save_freq='epoch',
                            verbose=1
                        )

                        log_saver = CSVLogger(
                            path_to_log_accuracy,
                            append=True, 
                            separator=','
                        )

                        model_saver_A = ModelCheckpoint(
                            f'models/best/best.h5', 
                            monitor='val_accuracy',
                            mode='max',
                            save_best_only=True,
                            save_freq='epoch',
                            verbose=1
                        )

                        model_saver_B = ModelCheckpoint(
                            f'models/{optimizer}/{batch}/best/best.h5', 
                            monitor='val_accuracy',
                            mode='max',
                            save_best_only=True,
                            save_freq='epoch',
                            verbose=1
                        )

                        model_saver_C =ModelCheckpoint(
                            f'models/{optimizer}/{batch}/{initializer_A[0]}/best/best.h5', 
                            monitor='val_accuracy',
                            mode='max',
                            save_best_only=True,
                            save_freq='epoch',
                            verbose=1
                        )

                        model_saver_D = ModelCheckpoint(
                            f'models/{optimizer}/{batch}/{initializer_A[0]}/{initializer_B[0]}/best/best.h5', 
                            monitor='val_accuracy',
                            mode='max',
                            save_best_only=True,
                            save_freq='epoch',
                            verbose=1
                        )

                        with tf.device('/GPU:0'):
                        
                            model = inception_resnet_v2(initializer_A[1], initializer_B[1], nbr_classes)

                            model.compile(
                                optimizer=opt, 
                                loss='categorical_crossentropy',
                                metrics='accuracy'
                            )
                        
                            model.fit(
                                train_generator,
                                validation_data=val_generator,
                                epochs=epochs,
                                batch_size=batch,
                                verbose=1,
                                callbacks=[model_saver_A, model_saver_B, model_saver_C, model_saver_D, 
                                weight_saver, log_saver]
                            )

                            model.save_weights(f'weights/{optimizer}/{batch}/{initializer_A[0]}/{initializer_B[0]}/{lr}/final.h5')
                        
                        with tf.device('/GPU:0'):

                            print('\n\n=== FINAL MODEL ===')

                            val_acc = model.evaluate(val_generator)
                            print(f'Validation accuracy: {val_acc}')

                            test_acc = model.evaluate(test_generator)
                            print(f'Test accuracy: {test_acc}')

                            write_accuracy_log(path_to_log_accuracy, val_acc, test_acc, '---FINAL MODEL---')
                            
                            model_evaluation_cmf(
                                model,
                                path_to_test, 
                                path_to_log_conf,
                                path_to_log_accuracy,
                                nbr_classes,
                                batch,
                                [initializer_A[0], initializer_B[0]],
                                'final'
                            )

                            print('\n\n=== BEST MODEL ===')
                            
                            model = inception_resnet_v2(initializer_A[1], initializer_B[1], nbr_classes)

                            model.load_weights(
                                f'weights/{optimizer}/{batch}/{initializer_A[0]}/{initializer_B[0]}/{lr}/best.h5'
                            )

                            model.compile(
                                optimizer=opt, 
                                loss='categorical_crossentropy',
                                metrics='accuracy'
                            )

                            val_acc = model.evaluate(val_generator)
                            print(f'Validation accuracy: {val_acc}')

                            test_acc = model.evaluate(test_generator)
                            print(f'Test accuracy: {test_acc}')

                            write_accuracy_log(path_to_log_accuracy, val_acc, test_acc, '---BEST MODEL---')

                            model_evaluation_cmf(
                                model,
                                path_to_test, 
                                path_to_log_conf,
                                path_to_log_accuracy,
                                nbr_classes,
                                batch,
                                [initializer_A[0], initializer_B[0]],
                                'best'
                            )

        df_sub = pd.read_csv(f'log/{optimizer}/{batch}/conf-{batch}.csv')
        max_accuracy = df_sub['accuracy'] == df_sub['accuracy'].max()
        df = df.append(df_sub.loc[max_accuracy])
        print(df_sub.loc[max_accuracy])
    
    df.to_csv('log/best_model.csv', index=False)
       




                    



                





        





