def prepend_dataset_with_weather(samples, location='Fresno, CA', weather_columns=None, use_cache=True, verbosity=0):
    """ Prepend weather the values specified (e.g. Max TempF) to the samples[0..N]['input'] vectors

    samples[0..N]['target'] should have an index with the date timestamp

    If you use_cache for the curent year, you may not get the most recent data.

    Arguments:
        samples (list of dict): {'input': np.array(), 'target': pandas.DataFrame}
    """
    if verbosity > 1:
        print('Prepending weather data for {} to dataset samples'.format(weather_columns))
    if not weather_columns:
        return samples
    timestamps = pd.DatetimeIndex([s['target'].index[0] for s in samples])
    years = range(timestamps.min().date().year, timestamps.max().date().year + 1)
    weather_df = weather.daily(location=location, years=years, use_cache=use_cache)
    # FIXME: weather_df.resample('D') fails
    weather_df.index = [d.date() for d in weather_df.index]
    if verbosity > 1:
        print('Retrieved weather for years {}:'.format(years))
        print(weather_df)
    weather_columns = [label if label in weather_df.columns else weather_df.columns[int(label)]
                       for label in (weather_columns or [])]
    for sampnum, sample in enumerate(samples):
        timestamp = timestamps[sampnum]
        try:
            weather_day = weather_df.loc[timestamp.date()]
        except:
            from traceback import print_exc
            print_exc()
            weather_day = {}
            if verbosity >= 0:
                warnings.warn('Unable to find weather for the date {}'.format(timestamp.date()))
        NaN = float('NaN')
        sample['input'] = [weather_day.get(label, None) for label in weather_columns] + list(sample['input'])
        if verbosity > 0 and NaN in sample['input']:
            warnings.warn('Unable to find weather features {} in the weather for date {}'.format(
                [label for i, label in enumerate(weather_columns) if sample['input'][i] == NaN], timestamp))
    return samples