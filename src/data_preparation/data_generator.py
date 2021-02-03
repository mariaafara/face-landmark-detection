import numpy as np
import random
import cv2
from math import sin, cos, pi
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    """ This Custom DataGenerator will load mini-batches and feed to the model dynamically.
    It can handle any number of coordinates.
    """

    def __init__(self, df, batch_size=32, rgb_channels=False,
                 shuffle=False, augment=True, normalize_image=False,
                 normalize_coordinates=False, scaler=None):
        """
        Args:

            df: Dataframe that we should generate batches from.
            batch_size: The size of the batch to be generated.
            rgb_channels: Boolean to specify if we want to repeat the grayscale image over three channels. Keras
            pre-trained models have trained on color images and if we want to use grayscale images we can still
            use these pre-trained models by repeating the grayscale image over three channels.
            shuffle: Option to shuffle the data.
            augment: Allow augmentation.
            normalize_image: Option to normalize the image.
            normalize_coordinates: Option to normalize the coordinates using MinMax scaler
            scaler: Normalize the coordinates using this scaler. It can only be provided if augment boolean is set to
            True.
            rotation_angles: A list with different rotation angles. Can only be provided if augment boolean is set to
            True.
            shift_pixels: A list of different shift pixels. Can only be provided if augment boolean is set to
            True.

        left_right_dic: To keep track on which pairs of landmarks to be swapped, we introduce a dictionary recording
        the original and new landmark's index.

        """
        self.df = df
        self.batch_size = batch_size
        self.size = df.shape[0]
        self.shuffle = shuffle
        self.augment = augment
        self.rgb_channels = rgb_channels
        self.normalize_image = normalize_image
        self.normalize_coordinates = normalize_coordinates
        self.scaler = scaler
        self.feature_columns = df.columns.tolist()[:-1]
        self.nb_features = len(self.feature_columns)
        self.left_right_dic = {'right_eye_center_x': 'left_eye_center_x',
                               'right_eye_center_y': 'left_eye_center_y',
                               'right_eye_inner_corner_x': 'left_eye_inner_corner_x',
                               'right_eye_inner_corner_y': 'left_eye_inner_corner_y',
                               'right_eye_outer_corner_x': 'left_eye_outer_corner_x',
                               'right_eye_outer_corner_y': 'left_eye_outer_corner_y',
                               'right_eyebrow_inner_end_x': 'left_eyebrow_inner_end_x',
                               'right_eyebrow_inner_end_y': 'left_eyebrow_inner_end_y',
                               'right_eyebrow_outer_end_x': 'left_eyebrow_outer_end_x',
                               'right_eyebrow_outer_end_y': 'left_eyebrow_outer_end_y',
                               'mouth_right_corner_x': 'mouth_left_corner_x',
                               'mouth_right_corner_y': 'mouth_left_corner_y'}
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.size // self.batch_size

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.size)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generates one batch of data."""
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batched_df = self.df.iloc[indexes]

        if self.augment:
            df_batch = self.augmentor(batched_df)
        else:
            df_batch = batched_df

        batched_images = np.vstack(df_batch["Image"].values).reshape(-1, 96, 96, 1).astype(np.float32)

        if self.rgb_channels:
            batched_images = np.repeat(batched_images, 3, -1)

        if self.normalize_image:
            batched_images = batched_images / 255

        batched_feature_coordinates = df_batch.iloc[:, :-1]
        if self.normalize_coordinates:
            batched_feature_coordinates = self.scaler.transform(batched_feature_coordinates.values)

        batched_feature_coordinates = batched_feature_coordinates.astype(np.float32)

        return batched_images, batched_feature_coordinates

    def augmentor(self, df):

        def flip_horizontally(row):
            """Method that flips the image horizontally and swaps the left and right features.
            Notes:
                 - y-coordinates values will stay the same while, the x-coordinates will have to be changed.
                 - Subtract our initial x-coordinate values from width of the image(96).
            """
            row["Image"] = np.flip(row["Image"], axis=1)
            for i in range(len(self.feature_columns)):
                if i % 2 == 0:  # Because there is always x-coordinate then y-coordinate.
                    row[self.feature_columns[i]] = 96. - row[self.feature_columns[i]]
            # swap
            rights = [a for a in self.feature_columns if "right" in a.split("_")]

            for col_r in rights:
                col_l = self.left_right_dic[col_r]
                temp = row[col_l]
                row[col_l] = row[col_r]
                row[col_r] = temp

            return row

        def increase_brightness(image):
            """Method that randomly increase image brightness.
            Notes:
                - Multiply pixel values by random value between 1 and 1.5 to increase the brightness of the image.
                - Clip the value between 0 and 255.
            """
            image = np.clip(random.uniform(1, 1.4) * image, 0.0, 255.0)
            return image

        def decrease_brightness(image):
            """Method that randomly decrease image brightness.
            Notes:
                - Multiply pixel values by random values between 0 and 0.1 to decrease the brightness of the image.
                - Clip the value between 0 and 255
            """
            image = np.clip(random.uniform(0.2, 0.7) * image, 0.0, 255.0)
            return image

        def shift(row):
            keypoint = row.iloc[:self.nb_features]
            image = row["Image"].astype(np.uint8)
            shift = np.random.randint(0, 20)
            (shift_x, shift_y) = random.choices([(-shift, -shift), (-shift, shift), (shift, -shift), (shift, shift)])[0]
            matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            shifted_image = cv2.warpAffine(image, matrix, (96, 96), flags=cv2.INTER_CUBIC)
            shifted_keypoint = np.array(
                [(point + shift_x) if idx % 2 == 0 else (point + shift_y) for idx, point in enumerate(keypoint)])
            if np.all(0.0 < shifted_keypoint) and np.all(shifted_keypoint < 96.0):
                # shifted_image = shifted_image.reshape(96, 96, 1)
                shifted_keypoint = np.clip(shifted_keypoint, 0.0, 96.0)
                row.iloc[:self.nb_features] = shifted_keypoint
                row["Image"] = shifted_image
                return row
            return row

        def rotate(row):
            keypoint = row.iloc[:self.nb_features]
            image = row["Image"].astype(np.uint8)
            angle_rand = np.random.randint(0, 20)
            angle = random.choices([angle_rand, -angle_rand])[0]
            middle = 96 / 2  # the middle value of the image dimension
            matrix = cv2.getRotationMatrix2D((middle, middle), angle, 1.0)
            angle_rad = -angle * pi / 180.  # Obtain angle in radians from angle in degrees
            # (notice negative sign for change in clockwise vs anti-clockwise directions from conventional rotation to
            # cv2's image rotation)
            rotated_image = cv2.warpAffine(image, matrix, (96, 96), flags=cv2.INTER_CUBIC)
            rotated_keypoint = keypoint - middle  # Subtract the middle value of the image dimension
            for idx in range(0, len(rotated_keypoint), 2):
                # https://in.mathworks.com/matlabcentral/answers/93554-how-can-i-rotate-a-set-of-points-in-a-plane-by-a-certain-angle-about-an-arbitrary-point
                rotated_keypoint[idx] = rotated_keypoint[idx] * cos(angle_rad) - rotated_keypoint[idx + 1] * sin(
                    angle_rad)
                rotated_keypoint[idx + 1] = rotated_keypoint[idx] * sin(angle_rad) + rotated_keypoint[
                    idx + 1] * cos(angle_rad)
            rotated_keypoint += middle  # Add the earlier subtracted value
            row.iloc[:self.nb_features] = rotated_keypoint
            row["Image"] = rotated_image
            return row

        def augment(row):
            # If we have more than one feature allow flipping.
            if len(self.feature_columns) > 2:
                if row["augment_flip"] == 1:
                    row = flip_horizontally(row)

            if row["augment_inc_b"]:
                row["Image"] = increase_brightness(row["Image"])

            if row["augment_dec_b"]:
                row["Image"] = decrease_brightness(row["Image"])

            # Rotation augmentation
            if row["augment_rotate"]:
                row = rotate(row)

            # shift horizontally or vertically by a random shift pixel value
            if row["augment_shift"]:
                row = shift(row)

            return row

        # Generate random boolean for augmentation types.
        bools = [True, False]
        weights = [0.6, 0.4]
        df["augment_flip"] = [random.getrandbits(1) for i in range(len(df))]
        df["augment_inc_b"] = [random.getrandbits(1) for i in range(len(df))]
        df["augment_dec_b"] = [random.getrandbits(1) for i in range(len(df))]
        df["augment_shift"] = [random.choices(bools, weights, k=1)[0] for i in range(len(df))]
        df["augment_rotate"] = [random.choices(bools, weights, k=1)[0] for i in range(len(df))]

        df = df.apply(augment, axis=1)

        df.drop(['augment_flip', 'augment_inc_b', 'augment_dec_b', "augment_shift", "augment_rotate"],
                axis=1, inplace=True)
        return df
