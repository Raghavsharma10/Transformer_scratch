def daily(location='Fresno, CA', years=1, use_cache=True, verbosity=1):
    """Retrieve weather for the indicated airport code or 'City, ST' string.

    >>> df = daily('Camas, WA', verbosity=-1)
    >>> 365 <= len(df) <= 365 * 2 + 1
    True

    Sacramento data has gaps (airport KMCC):
        8/21/2013 is missing from 2013.
        Whole months are missing from 2014.
    >>> df = daily('Sacramento, CA', years=[2013], verbosity=-1)
    >>> 364 <= len(df) <= 365
    True
    >>> df.columns
    Index([u'PST', u'Max TemperatureF', u'Mean TemperatureF', u'Min TemperatureF', u'Max Dew PointF', u'MeanDew PointF', u'Min DewpointF', ...
    """
    this_year = datetime.date.today().year
    if isinstance(years, (int, float)):
        # current (incomplete) year doesn't count in total number of years
        # so 0 would return this calendar year's weather data
        years = np.arange(0, int(years) + 1)
    years = sorted(years)
    if not all(1900 <= yr <= this_year for yr in years):
        years = np.array([abs(yr) if (1900 <= abs(yr) <= this_year) else (this_year - abs(int(yr))) for yr in years])[::-1]

    airport_code = airport(location, default=location)

    # refresh the cache each time the start or end year changes
    cache_path = 'daily-{}-{}-{}.csv'.format(airport_code, years[0], years[-1])
    cache_path = os.path.join(CACHE_PATH, cache_path)
    if use_cache:
        try:
            return pd.DataFrame.from_csv(cache_path)
        except:
            pass

    df = pd.DataFrame()
    for year in years:
        url = ('http://www.wunderground.com/history/airport/{airport}/{yearstart}/1/1/' +
               'CustomHistory.html?dayend=31&monthend=12&yearend={yearend}' +
               '&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&MR=1&format=1').format(
            airport=airport_code,
            yearstart=year,
            yearend=year
            )
        if verbosity > 1:
            print('GETing *.CSV using "{0}"'.format(url))
        buf = urllib.urlopen(url).read()
        if verbosity > 0:
            N = buf.count('\n')
            M = (buf.count(',') + N) / float(N)
            print('Retrieved CSV for airport code "{}" with appox. {} lines and {} columns = {} cells.'.format(
                  airport_code, N, int(round(M)), int(round(M)) * N))
            if verbosity > 2:
                print(buf)
        table = util.read_csv(buf, format='header+values-list', numbers=True)
        # # clean up the last column (if it contains <br> tags)
        table = [util.strip_br(row) if len(row) > 1 else row for row in table]
        # numcols = max(len(row) for row in table)
        # table = [row for row in table if len(row) == numcols]
        columns = table.pop(0)
        tzs = [s for s in columns if (s[1:] in ['ST', 'DT'] and s[0] in 'PMCE')]
        dates = [float('nan')] * len(table)
        for i, row in enumerate(table):
            for j, value in enumerate(row):
                if not value and value is not None:
                    value = 0
                    continue
                if columns[j] in tzs:
                    table[i][j] = util.make_tz_aware(value, tz=columns[j])
                    if isinstance(table[i][j], datetime.datetime):
                        dates[i] = table[i][j]
                        continue
                try:
                    table[i][j] = float(value)
                    if not (table[i][j] % 1):
                        table[i][j] = int(table[i][j])
                except:
                    pass
        df0 = pd.DataFrame(table, columns=columns, index=dates)
        df = df.append(df0)

    if verbosity > 1:
        print(df)

    try:
        df.to_csv(cache_path)
    except:
        if verbosity > 0 and use_cache:
            from traceback import print_exc
            print_exc()
            warnings.warn('Unable to write weather data to cache file at {}'.format(cache_path))

    return df