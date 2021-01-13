# Facial Landmarks Detection

In this project we aim at detecting the facial coordinates given a picture of a face.<br>

We want to detect the coordinates and orientation of eyes, eyebrows, nose and mouth in a picture.<br>
In the dataset at hand, we have 15 features (facial landmarks) that we want to predict, each has x and y coordinates.

    [left_eye_center, right_eye_center, left_eye_inner_corner, left_eye_outer_corner, right_eye_inner_corner, right_eye_outer_corner, left_eyebrow_inner_end, left_eyebrow_outer_end, right_eyebrow_inner_end, right_eyebrow_outer_end , nose_tip, mouth_left_corner, mouth_right_corner, mouth_center_top_lip, mouth_center_bottom_lip]

Here's an example: <br>

<img align="left" src="https://github.com/mariaafara/face-landmark-detection/blob/main/images/example.png">

<br>
<br>

  - **Eyes**: 3 coordinates (center, inner and outer part of the eye) will be predicted for both eyes (left and right eyes).
  - **Eyebrows**: 2 coordinates (inner and outer side of the eyebrow) will be detected for both eyebrows.
  - **Nose**: 1 coordinates which corresponds to the nose tip will be obtained.
  - **Mouth**: 4 coordinates (left, right, top and bottom part of the lip) will be predicted.
<br>

## Work Plan

The work is divided into 2 parts:
- **Part 1:** Data preparation 
- **Part 2:** Building the prediction model

### Part 1
 This part can also be divided to 4 sub-parts:
- **Part 1.1** EDA and Feature Engineering
- **Part 1.2** Create custom Data Generator
- **Part 1.3** Building a Prediction model for a feature (for each 14 feature)
- **Part 1.4** Label missing values

After performing some EDA, it turned out that 4909 examples (from total of 7049) contain missing values which means more than the half of the dataset. 
Since we cannot drop them, we have to find another alternative.

[comment]: <> (![]&#40;https://github.com/mariaafara/face-landmark-detection/blob/main/images/missing_data_nb.png&#41;)

<p float="left">
  <img src="/images/missing_data_nb.png"  />
  <img src="/images/missing_features_count.png"  /> 
</p>

Inorder to fill the missing values, we shall train a prediction model for each feature except the nose as it doesn't have any missing value (See 1st figure above). Thus, we shall train 14 models. 
Before doing that, we need to separate features and prepare the datasets inorder to train the models.

Then we need to build a custom Data Generator that will load mini-batches, perform data augmentation and normalization. It will be in handy for Part 1.3 and Part 2.

Part 1.1 and 1.2 are present in this [Notebook](https://github.com/mariaafara/face-landmark-detection/blob/main/data_preparation.ipynb).

In Part 1.3 we have trained a model for each feature except the nose_tip since it doesn't have missing value.

In Part 1.4 we have use the trained models to fill the missing values and prepare the main dataset inorder to use it for training the main prediction model that predicts all features.

Part 1.4 is present in this [Notebook](https://github.com/mariaafara/face-landmark-detection/blob/main/label_missing_values.ipynb).

Here are some examples of the final dataset:

<img align="left" src="https://github.com/mariaafara/face-landmark-detection/blob/main/images/final_examples.png">

