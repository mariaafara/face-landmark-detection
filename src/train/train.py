import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def euclidean_distance_loss(y_true, y_pred):
    """Custom loss function: Euclidean distance loss"""
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def absolute_tanh(x):
    """Custom activation function: Absolute tanh"""
    return K.abs(K.tanh(x))


def callbacks_functions(model_path, early_stopping_b=True, check_pointer_b=True, reduce_lr_b=True, patience=8):
    """define callbacks functions, save the best model with lower validation loss"""

    callbacks = []
    if early_stopping_b:
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
        callbacks.append(early_stopping)
    if check_pointer_b:
        check_pointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True, save_weights_only=True)
        callbacks.append(check_pointer)
    if reduce_lr_b:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience, verbose=1, mode='min')
        callbacks.append(reduce_lr)

    return callbacks


def compile_model(model, loss=euclidean_distance_loss, optimizer=Adam(learning_rate=1e-4), metrics=["mae"]):
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics = metrics
                  )
    return model


def fit_model(model, epochs, train_generator, val_generator, model_path, early_stopping_b=True, check_pointer_b=True, reduce_lr_b=True, patience=8):

    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=epochs,
                        callbacks=callbacks_functions(model_path, early_stopping_b, check_pointer_b, reduce_lr_b, patience)
                        )
    return model, history
