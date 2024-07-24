import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def proximo_id(file_path):
    df = pd.read_csv(file_path)
    if df.empty:
        id_rodada = 1
    else:
        id_rodada = df['id_rodada'].max() + 1
    
    novo_id = pd.DataFrame({'id_rodada': [id_rodada]})
    novo_id.to_csv(file_path, mode='a', header=False, index=False)
    
    return id_rodada



def avaliar_modelo(model, rede, id_rodada, dataset_teste, dados_teste, batch_size, imagenet, data_aug, nome_otimizador, nome_loss, lr, attention, camada_pooling, flatten, denses, dropouts, tempo_treino):
    nomes_metricas = ['Loss', 'Accuracy', 'Precision', 'Recall', 'AUC']
    dir_modelos_salvos_para_teste = f'./redes/{rede}/modelos_salvos'
    resultados_lista = []

    os.makedirs('./log_teste/previsoes', exist_ok=True)

    for nome_modelo in os.listdir(dir_modelos_salvos_para_teste):
        if nome_modelo.startswith(f'{rede}-{id_rodada}'):
            caminho_completo = os.path.join(dir_modelos_salvos_para_teste, f'{nome_modelo}')
            model.load_weights(caminho_completo)
            
            resultado_teste = model.evaluate(dataset_teste, steps=np.ceil(len(dados_teste) / batch_size))
            print(f'modelo {nome_modelo} - resultado: {resultado_teste}')
            
            resultados_teste_float = [float(x) for x in resultado_teste]
            
            densas_preenchidas = denses + [None] * (5 - len(denses))
            dropouts_preenchidos = dropouts + [None] * (5 - len(dropouts))

            chaves = ['arquivo', 'rede', 'imagenet', 'dataaug', 'otimizador', 'lossname', 'lr', 'attention', 'batchsize', 'pooling', 'flatten'] + \
                     [f'dense_{i+1}' for i in range(5)] + [f'dropout_{i+1}' for i in range(5)] + \
                     nomes_metricas + ['tempo_treino']
            valores = [nome_modelo, rede, imagenet, data_aug, nome_otimizador, nome_loss, lr, attention, batch_size, camada_pooling, flatten] + \
                      densas_preenchidas + dropouts_preenchidos + \
                      resultados_teste_float + [tempo_treino]

            resultados_dict = dict(zip(chaves, valores))

            resultados_lista.append(resultados_dict)
            
            salvar_previsoes(model, dataset_teste, dados_teste, nome_modelo, batch_size)
    
    salvar_resultados(resultados_lista)


def salvar_previsoes(model, dataset_teste, dados_teste, nome_modelo, batch_size):
    
    num_steps = np.ceil(len(dados_teste) / batch_size)
    predicoes = model.predict(dataset_teste, steps=num_steps)
    labels_teste = dados_teste[['burn_1', 'burn_2p', 'burn_2s', 'burn_3', 'burn_not']].values
    ids = dados_teste['id'].values

    resultados_df = pd.DataFrame({
        'id': ids, 
        'real': labels_teste.argmax(axis=1),  
        'predito': predicoes.argmax(axis=1)
    })
    
    resultados_df.to_csv(f'./log_teste/previsoes/resultados_{nome_modelo}.csv', index=False)


def calcular_metricas(nome_modelo):
    resultados_df = pd.read_csv(f'./log_teste/previsoes/resultados_{nome_modelo}.csv')
    y_true = resultados_df['real']
    y_pred = resultados_df['predito']
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    y_true_one_hot = pd.get_dummies(y_true).values
    y_pred_one_hot = pd.get_dummies(y_pred).reindex(columns=pd.get_dummies(y_true).columns, fill_value=0).values
    roc_auc = roc_auc_score(y_true_one_hot, y_pred_one_hot, average='weighted', multi_class='ovr')
    return acc, precision, recall, f1, roc_auc


def salvar_resultados(resultados_lista):
    log_teste_df = pd.DataFrame(resultados_lista)
    cabecalho = not os.path.isfile(f'./log_teste/log_teste.csv')
    log_teste_df.to_csv(f'./log_teste/log_teste.csv', mode='a', header=cabecalho, index=False)


def matriz_confusao(nome_modelo):
    resultados_df = pd.read_csv(f'./log_teste/previsoes/resultados_{nome_modelo}.csv')
    y_true = resultados_df['real']
    y_pred = resultados_df['predito']
    
    cm = confusion_matrix(y_true, y_pred)
    labels = ['burn_1', 'burn_2p', 'burn_2s', 'burn_3', 'burn_not']

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    np.set_printoptions(precision=2)
    print('matriz')
    print(cm_normalized)
    

    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap="Purples", xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    plt.xlabel('pred')
    plt.ylabel('real')
    plt.title(f'matriz - {nome_modelo}')
    plt.show()
