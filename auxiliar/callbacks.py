import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras import backend as K


    #########################
    ##                     ##
    ##      auxiliares     ##
    ##                     ##
    #########################


'''
usei so callback ja nativo do tf. callback é tipo umas funcoes ja feitas pra elas irem sendo chamadas enquanto treina.
utilizei 5 de 4 modelos diferentes.

argumentos:
    - dir_modelos_salvos: dir onde os modelos salvos serao armazenados.
    - dir_csv_log: dir onde o log CSV sera salvo.
    - check_best: flag para salvar o melhor modelo com base na validacao.
    - early_stop: flag para usar o EarlyStopping.
    - log: flag para salvar logs em um arquivo CSV.
    - reduce_lr: flag para reduzir a taxa de aprendizado quando a validação não melhorar.
    - check: flag para salvar modelos periodicamente.
    - early_stop_epocas: quantas epocas sem melhorar pra parar o treinamento.
    - check_epocas: numero de epocas para salvar modelos periodicamente.
    - reduce_lr_epocas: quantas epocas sem melhorar pra reduzir.
    - fator_reduce_lr: fator pelo qual a taxa de aprendizado será reduzida.

retorna:
    - callbacks: lista de callbacks configurados.
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
        print('check')
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
