

def separate_data_with_missing_values(main_df):
    """

    :param main_df: The original dataframe.
    :return:
    """

    dic_non_missing_data = {}
    dic_missing_data = {}

    cols = main_df.columns.tolist()[:30]
    #  Take out the nose feature.
    cols.remove('nose_tip_x')
    cols.remove('nose_tip_y')

    for col_i in range(0,len(cols),2):
        feature_name = cols[col_i][:-2]  # Taking the x and y coordinates of a certain feature.
        feature_df_cols = [cols[col_i], cols[col_i+1], "Image"]
        # Get all rows that does not contain a missing value for the specified feature.
        feature_df = main_df[feature_df_cols]

        df_feature_non_missing = feature_df[~feature_df.isnull().any(axis=1)]
        df_feature_missing = feature_df[feature_df.isnull().any(axis=1)]

        dic_non_missing_data[feature_name] = df_feature_non_missing
        dic_missing_data[feature_name] = df_feature_missing

    return dic_missing_data, dic_non_missing_data
