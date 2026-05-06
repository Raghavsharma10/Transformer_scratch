def dropna_columns(data: pd.DataFrame, max_na_values: int=0.15):
    """
    Remove columns with more NA values than threshold level

    :param data:
    :param max_na_values: proportion threshold of max na values
    :return:

    """
    size = data.shape[0]
    df_na = (data.isnull().sum()/size) >= max_na_values
    data.drop(df_na[df_na].index, axis=1, inplace=True)