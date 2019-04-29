import keras
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Sequential

def default(config):

    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                activation='relu',
                padding='same',
                kernel_regularizer=keras.regularizers.l2(0.),
                input_shape=(config.height, config.width, 5 * 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3),
                activation='relu',
                padding='same',
                kernel_regularizer=keras.regularizers.l2(0.)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3),
                activation='relu',
                padding='same',
                kernel_regularizer=keras.regularizers.l2(0.)))
    
    return model