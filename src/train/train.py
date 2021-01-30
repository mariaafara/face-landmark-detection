import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def euclidean_distance_loss(y_true, y_pred):
    """Custom loss function: Euclidean distance loss"""
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def absolute_tanh(x):
    """Custom activation function: Absolute tanh"""
    return K.abs(K.tanh(x))


def callbacks_functions(model_path, patience):
    """define callbacks functions, save the best model with lower validation loss"""
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

    check_pointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience, verbose=1, mode='min')

    return [check_pointer, early_stopping, reduce_lr]


def fit_model(model, epochs, train_generator, val_generator, model_path, patience=8):
    model.compile(loss=euclidean_distance_loss,
                  optimizer=Adam(learning_rate=1e-4),
                  )

    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=epochs,
                        callbacks=callbacks_functions(model_path, patience)
                        )
    return model, history
