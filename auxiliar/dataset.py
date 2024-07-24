import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_v3
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inception_resnet_v2
from keras.applications.efficientnet_v2 import preprocess_input as preprocess_efficientnet_v2s
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet50v2
from classification_models.keras import Classifiers



def get_preprocess(cnn):
    if cnn == 'ResNeXt50':
        ResNeXt50, preprocess_input = Classifiers.get('resnext50')
        return preprocess_input, (224, 224)
    elif cnn == 'ResNeXt101':
        ResNeXt101, preprocess_input = Classifiers.get('resnext101')
        return preprocess_input, (224, 224)
    
    preprocess_dict = {
        'InceptionV3': (preprocess_inception_v3, (299, 299)),
        'InceptionResNetV2': (preprocess_inception_resnet_v2, (299, 299)),
        'EfficientNetV2L': (preprocess_efficientnet_v2s, (480, 480)),
        'EfficientNetV2S': (preprocess_efficientnet_v2s, (384, 384)),
        'ResNet50V2': (preprocess_resnet50v2, (224, 224)),
        'ResNet152V2': (preprocess_resnet50v2, (224, 224)),
        'EfficientNetV2M': (preprocess_efficientnet_v2s, (480, 480)),
        'EfficientNetV2B3': (preprocess_efficientnet_v2s, (300, 300))
    }
    if cnn not in preprocess_dict:
        raise ValueError(f'nao tem esse modelo')
    return preprocess_dict[cnn]

data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(factor=0.15, fill_mode="nearest"),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="nearest"),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2, fill_mode="nearest"),
        layers.RandomFlip(mode="horizontal")
    ]
)

def processar_imagem(nome_do_arquivo, label, preprocess_input, img_tamanho):
    nome_da_imagem = tf.io.read_file(nome_do_arquivo)
    imagem_decodificada = tf.image.decode_jpeg(nome_da_imagem, channels=3)

    imagem_redimensionada = tf.image.resize(imagem_decodificada, img_tamanho)
    imagem_normalizada = preprocess_input(imagem_redimensionada)

    return imagem_normalizada, label

def criar_dataset(dataframe, diretorio, batch_size, rede, shuffle=False, repeat=True, data_aug=False):
    preprocess_input, img_tamanho = get_preprocess(rede)
    
    fotos = dataframe['id'].map(lambda x: f"{diretorio}/{x}").tolist()
    labels = dataframe[['burn_1', 'burn_2p', 'burn_2s', 'burn_3', 'burn_not']].values.tolist()
    
    dataset = tf.data.Dataset.from_tensor_slices((fotos, labels))
    
    def map_func(nome_do_arquivo, label):
        return processar_imagem(nome_do_arquivo, label, preprocess_input, img_tamanho)
    
    dataset = dataset.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    
    if data_aug:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda x, y: (tf.image.resize(x, img_tamanho), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    
    dataset = dataset.batch(batch_size)
    
    if repeat:
        dataset = dataset.repeat()
    
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset
