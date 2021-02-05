import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def save_to_pickle(object, file_path):
    pkl.dump(object, open(file_path, "wb"))


def load_from_pickle(file_path):
    return pkl.load(open(file_path, "rb"))


def split_data(df, test_size=0.2):
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=None)
    # print(df.shape, df_train.shape, df_test.shape)
    return df_train, df_test


def create_scaler(df):
    # fit the MinMaxScaler using the df data
    scaler = MinMaxScaler()
    xy = df.values
    scaler.fit_transform(xy)
    return scaler


def pre_process_images(df, rgb_channels, normalize_image):
    images = np.vstack(df["Image"].values).reshape(-1, 96, 96, 1).astype(np.float32)
    if rgb_channels:
        images = np.repeat(images,3,-1)
    if normalize_image == True:
        images = images/255
    return images


def pre_process_image(image, rgb_channels, normalize_image):
    if rgb_channels:
        image = np.repeat(image, 3, -1)
    if normalize_image:
        image = image / 255
    return image


def visualize_random_indexed_image(df, i=-1, image_path=None):
    """
    Method that visualize a certain image with index i in the df or a random image with i=-1.
    With plotting the facial key points specified in the df cols.
    Notes:
        - x-coordinates are in even columns like 0,2,4,.. and y-coordinates are in odd columns like 1,3,5,..
    """
    if i == -1:
        i = np.random.randint(1, len(df))

    plt.figure()
    plt.imshow(df['Image'][i], cmap='gray')

    nbr_cols = df.columns.tolist()[:-1]
    for j in range(0, len(nbr_cols), 2):
        plt.plot(df.loc[i][j], df.loc[i][j + 1], 'rx')

    if image_path is not None:
        plt.savefig(image_path)


def visualize_random_indexed_images(df, image_path):
    plt.figure(figsize=(30, 30))
    for i in range(12):
        plt.subplot(6, 6, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        j = np.random.randint(1, len(df))
        plt.imshow(df['Image'][j], cmap='gray')
        nbr_cols = df.columns.tolist()[:-1]
        for k in range(0, len(nbr_cols), 2):
            plt.plot(df.loc[j][k], df.loc[j][k + 1], 'X', color="blue")
    plt.savefig(image_path, bbox_inches="tight")
    plt.show()


def visualize_image(image, coordinates, image_path=None, predicted_coordinates=None):
    plt.figure()
    plt.imshow(image, cmap='gray')
    f = len(coordinates)
    for j in range(0, f, 2):
        plt.plot(coordinates[j], coordinates[j + 1], 'X', color='blue')
        if predicted_coordinates is not None:
            plt.plot(predicted_coordinates[j], predicted_coordinates[j + 1], 'X', color='red')
    plt.legend({'actual', 'predicted'})
    if image_path is not None:
        plt.savefig(image_path)


def visualize_random_predicted_images(my_model, batch, scaler, feature_name=None, image_path=None, type_="augmented"):

    fig = plt.figure(figsize=(30, 30))
    indxs = []
    for i in range(12):
        plt.subplot(6, 6, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        j= np.random.randint(0,len(batch[0]))

        while j in indxs:
          j= np.random.randint(0,len(batch[0]))

        indxs.append(j)

        plt.imshow(batch[0][j:j+1].reshape(96, 96, 1)[:,:,0], cmap='gray')

        predicted_y = my_model.predict(batch[0][j:j+1])
        predicted_y = scaler.inverse_transform(predicted_y)[0]

        nbr_cols = len(predicted_y)

        actual_y = batch[1][j:j + 1][0]
        actual_y = scaler.inverse_transform([actual_y])[0]
        for k in range(0, nbr_cols, 2):
            plt.plot(actual_y[k], actual_y[k + 1], 'X', color='blue')

        for k in range(0, nbr_cols, 2):
            plt.plot(predicted_y[k], predicted_y[k + 1], 'X', color='red')

    plt.legend({"actual", "predicted"})
    if feature_name:
        fig.suptitle("Trained {} feature model predictions on {} images from Test set".format(feature_name.upper(),type_), fontweight='bold', fontsize=30)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    if image_path:
        fig.savefig(image_path, bbox_inches="tight")
    plt.show()