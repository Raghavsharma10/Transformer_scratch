def cat_data(data_kw):
    """
    Concatenate data with ticker as sub column index

    Args:
        data_kw: key = ticker, value = pd.DataFrame

    Returns:
        pd.DataFrame

    Examples:
        >>> start = '2018-09-10T10:10:00'
        >>> tz = 'Australia/Sydney'
        >>> idx = pd.date_range(start=start, periods=6, freq='min').tz_localize(tz)
        >>> close_1 = [31.08, 31.10, 31.11, 31.07, 31.04, 31.04]
        >>> vol_1 = [10166, 69981, 14343, 10096, 11506, 9718]
        >>> d1 = pd.DataFrame(dict(price=close_1, volume=vol_1), index=idx)
        >>> close_2 = [70.81, 70.78, 70.85, 70.79, 70.79, 70.79]
        >>> vol_2 = [4749, 6762, 4908, 2002, 9170, 9791]
        >>> d2 = pd.DataFrame(dict(price=close_2, volume=vol_2), index=idx)
        >>> sample = cat_data({'BHP AU': d1, 'RIO AU': d2})
        >>> sample.columns
        MultiIndex(levels=[['BHP AU', 'RIO AU'], ['price', 'volume']],
                   codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
                   names=['ticker', None])
        >>> r = sample.transpose().iloc[:, :2]
        >>> r.index.names = (None, None)
        >>> r
                       2018-09-10 10:10:00+10:00  2018-09-10 10:11:00+10:00
        BHP AU price                       31.08                      31.10
               volume                  10,166.00                  69,981.00
        RIO AU price                       70.81                      70.78
               volume                   4,749.00                   6,762.00
    """
    if len(data_kw) == 0: return pd.DataFrame()
    return pd.DataFrame(pd.concat([
        data.assign(ticker=ticker).set_index('ticker', append=True)
            .unstack('ticker').swaplevel(0, 1, axis=1)
        for ticker, data in data_kw.items()
    ], axis=1))