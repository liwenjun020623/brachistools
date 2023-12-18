from turtle import pd

from keras.preprocessing.image import ImageDataGenerator
import Process.data_processing as pro
import network
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import tensorflow as tf


def train(aug, model, x_train, y_train, x_validation, y_validation):
    cbs = [ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.2, min_lr=1e-7),
           ModelCheckpoint("../Predict/model.h5", monitor='val_accuracy', save_best_only=True, mode='max'),
           TensorBoard(log_dir="../Predict", histogram_freq=1, write_grads=True)]
    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam", metrics=["accuracy"])  # Configuration
    history = model.fit(aug.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_validation, y_validation), steps_per_epoch=len(x_train) // batch_size,
                        callbacks=cbs, epochs=epochs, verbose=1)

    history_df = pd.DataFrame(history.history)
    history_df[['loss', 'val_loss']].plot()

    history_df = pd.DataFrame(history.history)
    history_df[['accuracy', 'val_accuracy']].plot()


if __name__ == "__main__":
    class_num = 2
    batch_size = 16
    epochs = 60
    with tf.device("/GPU:0"):
        model = network.ResNet50(classes=class_num)  # Load model
        x_train, y_train = pro.x_train, pro.y_train
        x_validation, y_validation = pro.x_validation, pro.y_validation  # Generate data
        aug = ImageDataGenerator(rotation_range=90, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.2, zoom_range=2,
                                 horizontal_flip=True, vertical_flip=True, fill_mode="nearest")  # Data enhancement

        train(aug, model, x_train, y_train, x_validation, y_validation)  # Train
