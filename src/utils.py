import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def save_to_pickle(object, file_path):
    pkl.dump(object, open(file_path, "wb"))


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


def visualize_random_indexed_images(df, image_path=None):
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
            plt.plot(df.loc[j][k], df.loc[j][k + 1], 'rx')
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
