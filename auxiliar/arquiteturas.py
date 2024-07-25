import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, ResNet152V2, ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Dense, Flatten, Input
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, AUC
from .attention_blocks import cbam_block, squeeze_excite_block
from efficientnet_v2 import EfficientNetV2S, EfficientNetV2M, EfficientNetV2B3, EfficientNetV2L
from classification_models.keras import Classifiers

'''
    retorna o modelo base e a funcao de preprocessamento para o modeloe scolhido
    
    argumentos:
        - nome: nome do modelo
        - weights: pesos pre-treinados a serem usados
        - include_top: flag indicando se inclui a parte superior da rede (sempre falso)
        
    retorna:
        - modelo base especificado e a funcao de preprocessamento correspondente(se for ResNeXt), se for as outras, so o modelo
'''

def get_model(nome, weights, include_top=False):
    if nome == 'ResNeXt50':
        ResNeXt50, preprocess_input = Classifiers.get('resnext50')
        return ResNeXt50(include_top=include_top, weights=weights, input_shape=(224, 224, 3)), preprocess_input
    elif nome == 'ResNeXt101':
        ResNeXt101, preprocess_input = Classifiers.get('resnext101')
        return ResNeXt101(include_top=include_top, weights=weights, input_shape=(224, 224, 3)), preprocess_input
    
    models_dict = {
        'EfficientNetV2S': EfficientNetV2S,
        'EfficientNetV2L': EfficientNetV2L,
        'InceptionV3': InceptionV3,
        'InceptionResNetV2': InceptionResNetV2,
        'EfficientNetV2M': EfficientNetV2M,
        'EfficientNetV2B3': EfficientNetV2B3,
        'ResNet152V2': ResNet152V2,
        'ResNet50V2': ResNet50V2
    }
    return models_dict[nome](weights=weights, include_top=include_top), None


'''
constroi e compila o modelo completo.  camadas densas, de dropout, atencao, etc etc e tal

argumentos:
    - base_model: modelo base pre-treinado(ou nao)
    - loss: funcao de perda
    - otimizador: otimizador
    - attention: tipo de bloco de atencao a ser usado ('se' ou 'cbam')
    - denses: lista com o numero de unidades em cada camada densa
    - dropouts: lista com as taxas de dropout correspondentes para cada camada densa
    - pooling: tipo de camada de pooling a ser usada ('global_max', 'global_avg')
    - flatten: flag para adicionar uma camada Flatten
    - funcao_atv: funcao de ativacao nas camadas densas (default: 'relu')
    - num_classes: numero de classes para classificacao (default: 5). as 5 classes, né? classificação 99% vai ser assim
    
retorna:
    - modelo compilado
'''

def build_model(
    base_model,
    loss,
    otimizador,
    attention,
    denses,
    dropouts,
    pooling,
    flatten=False,
    funcao_atv='relu',
    num_classes=5):

    x = base_model.output

    if attention == 'se':
        x = squeeze_excite_block(x)
    elif attention == 'cbam':
        x = cbam_block(x)

    if pooling == 'global_max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)
    elif pooling == 'global_avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    if flatten:
        x = Flatten(name='flatten')(x)

    for i, (camadas, dropout_rate) in enumerate(zip(denses, dropouts)):
        if camadas > 0:
            x = Dense(camadas, activation=funcao_atv, name=f'dense_{i+1}')(x)
            if dropout_rate > 0:
                x = Dropout(dropout_rate, name=f'dropout_{i+1}')(x)

    camada_de_classificacao = Dense(num_classes, activation='softmax', name='dense_class')(x)

    model = Model(inputs=base_model.input, outputs=camada_de_classificacao)
    model.compile(
        optimizer=otimizador,
        loss=loss,
        metrics=[
            CategoricalAccuracy(),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]
    )

    return model

'''
constroi um modelo customizado, carregando o modelo base especificado e ajustando conforme os parametros fornecidos

argumentos:
    - rede: nome do modelo base
    - loss: funcao de perda
    - weights: pesos pre-treinados. aqui eu to falando se eu importo com imgnet1k,21k ou só o modelo com pesos zerados
    - otimizador: otimizador
    - attention: tipo de bloco de atencao a ser usado ('se' ou 'cbam')
    - denses: lista com o numero de unidades em cada camada densa
    - dropouts: lista com as taxas de dropout correspondentes para cada camada densa
    - pooling: tipo de camada de pooling a ser usada ('global_max', 'global_avg')
    - flatten: flag para adicionar uma camada Flatten
    
retorna:
    - modelo compilado
'''

def custom_model(
        rede, 
        loss, 
        weights, 
        otimizador, 
        attention, 
        denses, 
        dropouts, 
        pooling, 
        flatten):
    base_model, preprocess_input = get_model(rede, weights)
    
    if not rede.startswith('ResNeXt'):
        base_model.trainable = True
    
    return build_model(base_model, loss, otimizador, attention, denses, dropouts, pooling, flatten)


