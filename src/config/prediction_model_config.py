from os import getenv

TEST_VAR = getenv("TEST_VAR", "DEFAULT_TEST_VAR")
DATA_LOAD_PATH = getenv("DATA_LOAD_PATH", "data/main_data2.pkl")
TRAIN_SCALER_PATH = getenv("TRAIN_SCALER_PATH", "scalers/main_scaler.pkl")
BATCH_SIZE = int(getenv("BATCH_SIZE", 32))
NORMALIZE_IMAGE = getenv("NORMALIZE_IMAGE", "true") == "true"
NORMALIZE_COORDINATES = getenv("NORMALIZE_COORDINATES", "true") == "true"
AUGMENT_TRAIN = getenv("AUGMENT_TRAIN", "true") == "true"
AUGMENT_VAL = getenv("AUGMENT_VAL", "false") == "true"
RGB_CHANNELS = getenv("RGB_CHANNELS", "true") == "true"
EPOCHS = int(getenv("EPOCHS", 2))
PATIENCE_EARLY_STOPPING = int(getenv("PATIENCE", 8))
PATIENCE_LR = int(getenv("PATIENCE", 4))
MODEL_PATH = getenv("MODEL_PATH", "models/prediction_model/model")  # .h5
WEIGHT_PATH = getenv("WEIGHT_PATH", "models/prediction_model/model_weights.ckpt")
IMAGE_SHAPE = tuple([int(i) for i in getenv("IMAGE_SHAPE", "96,96,3").split(",")])  #  ex: 96,96,3
BASE_MODEL_NAME = getenv("BASE_MODEL_NAME", 'VGG16')   # EfficientNetB0, InceptionV3, VGG16, VGG19
FEATURES_DIMENSIONS = int(getenv("FEATURES_DIMENSIONS", 2))  # 2 or 30
TRAINABLE = getenv("TRAINABLE", "false") == "true"
INTERMEDIATE_DIMENSIONS = int(getenv("INTERMEDIATE_DIMENSIONS", 64))
DROPOUT_RATE = float(getenv("DROPOUT_RATE", 0.4))
