def seasonal_subset(dataframe,
                    months='all'):
    '''Get the seasonal data.

    Parameters
    ----------
    dataframe : pd.DataFrame
    months: int, str
        Months to use for statistics, or 'all' for 1-12 (default='all')
    '''

    if isinstance(months, str) and months == 'all':
        months = np.arange(12) + 1

    for month_num, month in enumerate(months):
        df_cur = dataframe[dataframe.index.month == month]

        if month_num == 0:
            df = df_cur
        else:
            df = df.append(df_cur)

    return df.sort_index()