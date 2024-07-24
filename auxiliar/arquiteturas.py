import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, ResNet152V2, ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, AveragePooling2D, Dropout, Dense, Flatten, Input
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, AUC
from tensorflow.keras.losses import CategoricalCrossentropy
from .attention_blocks import cbam_block, squeeze_excite_block
from efficientnet_v2 import EfficientNetV2S, EfficientNetV2M, EfficientNetV2B3, EfficientNetV2L
from classification_models.keras import Classifiers

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

    camada_de_previsao = Dense(num_classes, activation='softmax', name='dense_pred')(x)

    model = Model(inputs=base_model.input, outputs=camada_de_previsao)
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

def custom_model(rede, loss, weights, otimizador, attention, denses, dropouts, pooling, flatten):
    base_model, preprocess_input = get_model(rede, weights)
    
    if not rede.startswith('ResNeXt'):
        base_model.trainable = True
    
    return build_model(base_model, loss, otimizador, attention, denses, dropouts, pooling, flatten)


