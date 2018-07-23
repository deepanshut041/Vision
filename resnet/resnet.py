from cifar10 import *
from keras.layers import *
from keras.models import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import pylab as pl

# get_dataset()
x_train, _, y_train= load_training_data()
x_test, _, y_test = load_test_data()
input_shape = list(x_train.shape[1:])
classes = y_train.shape[1]

def plot_results(results):
    pl.figure()

    pl.subplot(121)
    pl.plot(results.history['acc'])
    pl.title('Accuracy:')
    pl.plot(results.history['val_acc'])
    pl.legend(('Train', 'Validation'))

    pl.subplot(122)
    pl.plot(results.history['loss'])
    pl.title('Cost:')
    pl.plot(results.history['val_loss'])
    pl.legend(('Train', 'Validation'))

def building_block(X, filter_size, filters, stride=1):
    X_shortcut = X
    if stride > 1:
        X_shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)
    X = Conv2D(filters, kernel_size = filter_size, strides=stride, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters, kernel_size = filter_size, strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = add([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def create_model(input_shape, classes, name):
    X_input = Input(input_shape)
    X = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same')(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)

    X = building_block(X, filter_size=3, filters=32, stride=2)  # dimensions change (stride=2)
    X = building_block(X, filter_size=3, filters=32, stride=1)
    X = building_block(X, filter_size=3, filters=32, stride=1)
    X = building_block(X, filter_size=3, filters=32, stride=1)
    X = building_block(X, filter_size=3, filters=32, stride=1)

    X = building_block(X, filter_size=3, filters=64, stride=2)  # dimensions change (stride=2)
    X = building_block(X, filter_size=3, filters=64, stride=1)
    X = building_block(X, filter_size=3, filters=64, stride=1)
    X = building_block(X, filter_size=3, filters=64, stride=1)
    X = building_block(X, filter_size=3, filters=64, stride=1)

    X = GlobalAveragePooling2D()(X)
    X = Dense(classes, activation='softmax')(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name=name)

    return model


# Define optimizer and compile model
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
r_net = create_model(input_shape=input_shape, classes=classes, name='r_net')
r_net.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])


# Generator for data augmantation
datagen = ImageDataGenerator(
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True)  # randomly flip images


# Train model
results = r_net.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size = 250),
                                 epochs = 1,
                                 steps_per_epoch=200,  # data_size/batch size
                                 validation_data=(x_test, y_test))

# Plot train / validation results
r_net.save_weights("r_net_cifar_10.h5")
plot_results(results)

# Print model architecture
r_net.summary()


