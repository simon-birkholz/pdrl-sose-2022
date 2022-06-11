import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, ReLU, Add, BatchNormalization, GlobalAveragePooling2D, Dense, MaxPooling2D
from tqdm.keras import TqdmCallback


def resize_and_cast(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.image.resize(x, (128, 128))
    # x = tf.image.resize(x, (16, 16))
    return x, y


def res_block(input, n_filters, shortcut=False):
    conv = Conv2D(filters=n_filters, kernel_size=3, padding='same', strides=(1, 1))(input)
    conv = BatchNormalization()(conv)
    conv = ReLU()(conv)

    conv = Conv2D(filters=n_filters, kernel_size=3, padding='same', strides=(1, 1))(conv)
    conv = BatchNormalization()(conv)

    x = input
    if shortcut:
        x = Conv2D(filters=n_filters, kernel_size=1, padding='same', strides=(1, 1))(x)

    y = Add()([x, conv])
    y = ReLU()(y)

    return y


def resnet(input):
    x = Conv2D(64, kernel_size=(7, 7), activation='relu', padding='same', strides=(2, 2))(input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(strides=(2, 2))(x)
    x = res_block(x, 64)
    x = res_block(x, 64)
    x = res_block(x, 64)
    x = res_block(x, 128, True)
    x = res_block(x, 128)
    x = res_block(x, 128)
    x = res_block(x, 128)
    x = res_block(x, 256, True)
    x = res_block(x, 256)
    x = res_block(x, 256)
    x = res_block(x, 256)
    x = res_block(x, 512, True)
    x = res_block(x, 512)
    x = res_block(x, 512)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)
    return x


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')

    builder = tfds.builder('imagenette/160px-v2')
    builder.download_and_prepare()
    raw_train_ds, raw_valid_ds = builder.as_dataset(['train', 'validation'], as_supervised=True)
    train_ds = raw_train_ds.map(resize_and_cast).shuffle(10000).batch(256).prefetch(4)
    valid_ds = raw_valid_ds.map(resize_and_cast).batch(256).prefetch(4)

    inputs = Input(shape=(None, None, 3))
    outputs = resnet(inputs)
    model = Model(inputs=inputs, outputs=outputs, name='ResNet', )
    model.summary()

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    epochs = 5
    history = model.fit(train_ds, validation_data=valid_ds, epochs=epochs, verbose=1)
    test_loss, test_acc = model.evaluate(valid_ds, verbose=1)
    print(f'Test accuracy {test_acc}')
