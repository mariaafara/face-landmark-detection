from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPool2D, Conv2D


class MySimpleModel2(Model):
    def __init__(self, image_shape, features_dim):
        super(MySimpleModel2, self).__init__()

        self.image_shape = image_shape
        self.features_dim = features_dim
        self.conv1 = Conv2D(64, (3, 3), padding="same", )
        self.conv11 = Conv2D(64, (3, 3), padding="same", )
        self.pool1 = MaxPool2D((3, 3))
        self.conv2 = Conv2D(128, (3, 3), padding="same")
        self.conv22 = Conv2D(128, (3, 3), padding="same")
        self.pool2 = MaxPool2D((3, 3))
        self.conv3 = Conv2D(128, (3, 3), padding="same")
        self.pool3 = MaxPool2D((3, 3))
        self.conv4 = Conv2D(128, (3, 3), padding="same")
        self.pool4 = MaxPool2D((3, 3))
        self.dropout1 = Dropout(0.4)
        self.hidden1 = Dense(1024)
        self.flatten = Flatten()
        self.dropout2 = Dropout(0.4)
        self.hidden2 = Dense(512)
        self.hidden3 = Dense(self.features_dim)

        self.build((None,self.image_shape[0], self.image_shape[1], self.image_shape[2]))

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.conv11(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv22(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.dropout2(x)
        x = self.hidden3(x)
        return x