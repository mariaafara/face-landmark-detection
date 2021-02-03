from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPool2D, Conv2D, Input


class MySimpleModel(Model):
    def __init__(self, features_dim):
        super(MySimpleModel, self).__init__()

        self.feature_dim = features_dim
        self.conv1 = Conv2D(32, (3, 3), padding="same", )
        self.pool1 = MaxPool2D((3, 3))
        self.conv2 = Conv2D(64, (3, 3), padding="same")
        self.pool2 = MaxPool2D((3, 3))
        self.conv3 = Conv2D(64, (3, 3), padding="same")
        self.pool3 = MaxPool2D((3, 3))
        self.conv4 = Conv2D(64, (3, 3), padding="same")
        self.pool4 = MaxPool2D((3, 3))
        self.flatted = Flatten()
        self.dropout = Dropout(0.4)
        self.hidden1 = Dense(512)
        self.output = Dense(self.features_dim)

        def call(self, inputs):
            x = self.conv1(inputs)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.pool3(x)
            x = self.conv4(x)
            x = self.pool4(x)
            x = self.dropout(x)
            x = self.hidden1(x)
            x = self.output(x)
            return x
