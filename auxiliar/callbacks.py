import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras import backend as K


    #########################
    ##                     ##
    ##      auxiliares     ##
    ##                     ##
    #########################


'''
callbacks sao algumas funcoes ja estabelecidas pelo keras que executam acoes enquanto o modelo treina. (podemos fazer uns customizados)

1. `ModelCheckpoint`: salva os pesos do modelo durante o treinamento. configurei 2x. para salvar o melhor modelo (`save_best_only=True`) e periodicamente, no caso em 10 epocas (usando o parametro `period`).
2. `EarlyStopping`: interrompe o treinamento se a perda de validacao nao melhorar apos um numero especifico de epocas (`patience`). pra nao overfittar e nao passar horas em treinamento que nao vai melhorar
3. `ReduceLROnPlateau`: reduz a taxa de aprendizado quando a perda de validacao para de melhorar. as vezes o passo é grande demais e acaba preso sem conseguir achar o minimo
4. `CSVLogger`: salva os resultados de cada epoca em um csv
'''

def callbacks(dir_modelos_salvos, dir_csv_log, check_best, early_stop, log, reduce_lr, check, early_stop_epocas, check_epocas, reduce_lr_epocas, fator_reduce_lr):
    callbacks = []
    print("callbacks adicionados:") 

    if check_best:
        print('check best')
        checkpoint_best = ModelCheckpoint(
            monitor='val_loss',
            filepath=dir_modelos_salvos,
            save_weights_only=False,
            save_best_only=True,
            verbose=1,
        )
        callbacks.append(checkpoint_best)
        
    if check:
        print('check 10')
        checkpoint_15 = ModelCheckpoint(
            monitor='val_loss',
            filepath=dir_modelos_salvos,
            save_weights_only=False,
            save_best_only=False,
            verbose=1,
            period=check_epocas
        )
        callbacks.append(checkpoint_15)

    if early_stop:
        print('early stop')
        early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=early_stop_epocas,
            verbose=1,
            min_delta=0.01
        )
        callbacks.append(early_stopping)

    if log:
        print('log')
        csv_log = CSVLogger(
            dir_csv_log,
            append=True
        )
        callbacks.append(csv_log)

    if reduce_lr:
        print('reduzindo lr')
        reduce_learning_rate = ReduceLROnPlateau(
            monitor='val_loss',
            factor=fator_reduce_lr,
            patience=reduce_lr_epocas,
            min_lr=0.00000001,
            verbose=1
        )
        callbacks.append(reduce_learning_rate)

    return callbacks
