from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_input_inception_v3
from tensorflow.keras.applications.efficientnet import EfficientNetB0, \
    preprocess_input as preprocess_input_efficientnetB0


class BaseModel(layers.Layer):
    """Extracts features from the input image"""

    def __init__(self, image_shape, name, trainable=False):
        super(BaseModel, self).__init__(name=name)
        if name == "EfficientNetB0":
            self.base_model = EfficientNetB0(include_top=False, input_shape=image_shape)
            self.preprocess_input = preprocess_input_efficientnetB0
        elif name == "InceptionV3":
            self.base_model = InceptionV3(include_top=False, input_shape=image_shape)
            self.preprocess_input = preprocess_input_inception_v3
        elif name == "VGG16":
            self.base_model = VGG16(include_top=False, input_shape=image_shape)
            self.preprocess_input = preprocess_input_vgg16
        elif name == "VGG19":
            self.base_model = VGG19(include_top=False, input_shape=image_shape)
            self.preprocess_input = preprocess_input_vgg19

        self.base_model.trainable = trainable

        # set the first 15 layers (up to the last conv block)
        # to non-trainable (weights will not be updated)
        # for layer in self.base_model.layers[:15]:
        #     layer.trainable = False

    def call(self, inputs):
        x = self.preprocess_input(inputs)
        base_model_output = self.base_model(x)
        return base_model_output

class TopModel(layers.Layer):
    """Fully connected layers"""

    def __init__(self, features_dim, intermediate_dim=64, dropout_rate=0.4):
        super(TopModel, self).__init__()
        # self.dense1 = Dense(intermediate_dim, activation=relu)
        self.dropout = Dropout(rate=dropout_rate)
        self.dense2 = Dense(features_dim, activation=relu)
        self.global_average_pooling = GlobalAveragePooling2D()
        self.batch_normalization = BatchNormalization()

    def call(self, inputs):
        x = self.global_average_pooling(inputs)
        x = self.batch_normalization(x)
        # x = self.dense1(inputs)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class MyModel(Model):
    """Combines base and top models into an end-to-end model for training"""

    def __init__(self, image_shape, base_model_name, features_dim,
                 trainable=False, intermediate_dim=64, dropout_rate=0.4):
        super(MyModel, self).__init__()
        self.image_shape = image_shape
        self.base_model = BaseModel(image_shape, name=base_model_name, trainable=trainable)
        # self.flatten = Flatten()
        self.top_model = TopModel(features_dim=features_dim,
                                  intermediate_dim=intermediate_dim,
                                  dropout_rate=dropout_rate)
        self.build(self.base_model.base_model.input_shape)


    def call(self, inputs):
        x = self.base_model(inputs)
        # x = self.flatten(x)
        x = self.top_model(x)
        return x
