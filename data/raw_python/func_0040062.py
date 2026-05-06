def hourly(location='Fresno, CA', days=1, start=None, end=None, years=1, use_cache=True, verbosity=1):
    """ Get detailed (hourly) weather data for the requested days and location

    The Weather Underground URL for Fresno, CA on 1/1/2011 is:
    http://www.wunderground.com/history/airport/KFAT/2011/1/1/DailyHistory.html?MR=1&format=1

    This will fail periodically on Travis, b/c wunderground says "No daily or hourly history data available"
    >> df = hourly('Fresno, CA', verbosity=-1)
    >> 1 <= len(df) <= 24 * 2
    True
    The time zone of the client where this is used to compose the first column label, hence the ellipsis
    >> df.columns  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    Index([u'Time...

    >> df = hourly('Fresno, CA', days=5, verbosity=-1)
    >> 24 * 4 <= len(df) <= 24 * (5 + 1) * 2
    True
    """
    airport_code = airport(location, default=location)

    if isinstance(days, int):
        start = start or None
        end = end or datetime.datetime.today().date()
        days = pd.date_range(start=start, end=end, periods=days)

    # refresh the cache each calendar month or each change in the number of days in the dataset
    cache_path = 'hourly-{}-{}-{:02d}-{:04d}.csv'.format(airport_code, days[-1].year, days[-1].month, len(days))
    cache_path = os.path.join(CACHE_PATH, cache_path)
    if use_cache:
        try:
            return pd.DataFrame.from_csv(cache_path)
        except:
            pass

    df = pd.DataFrame()
    for day in days:
        url = ('http://www.wunderground.com/history/airport/{airport_code}/{year}/{month}/{day}/DailyHistory.html?MR=1&format=1'.format(
               airport_code=airport_code,
               year=day.year,
               month=day.month,
               day=day.day))
        if verbosity > 1:
            print('GETing *.CSV using "{0}"'.format(url))
        buf = urllib.urlopen(url).read()
        if verbosity > 0:
            N = buf.count('\n')
            M = (buf.count(',') + N) / float(N)
            print('Retrieved CSV for airport code "{}" with appox. {} lines and {} columns = {} cells.'.format(
                  airport_code, N, int(round(M)), int(round(M)) * N))
        if (buf.count('\n') > 2) or ((buf.count('\n') > 1) and buf.split('\n')[1].count(',') > 0):
            table = util.read_csv(buf, format='header+values-list', numbers=True)
            columns = [s.strip() for s in table[0]]
            table = table[1:]
            tzs = [s[4:] for s in columns if (s[5:] in ['ST', 'DT'] and s[4] in 'PMCE' and s[:4].lower() == 'time')]
            if tzs:
                tz = tzs[0]
            else:
                tz = 'UTC'
            for rownum, row in enumerate(table):
                try:
                    table[rownum] = [util.make_tz_aware(row[0], tz)] + row[1:]
                except ValueError:
                    pass
            dates = [row[-1] for row in table]
            if not all(isinstance(date, (datetime.datetime, pd.Timestamp)) for date in dates):
                dates = [row[0] for row in table]
            if len(columns) == len(table[0]):
                df0 = pd.DataFrame(table, columns=columns, index=dates)
                df = df.append(df0)
            elif verbosity >= 0:
                msg = "The number of columns in the 1st row of the table:\n    {}\n    doesn't match the number of column labels:\n    {}\n".format(
                    table[0], columns)
                msg += "Wunderground.com probably can't find the airport: {} ({})\n    or the date: {}\n    in its database.\n".format(
                    airport_code, location, day)
                msg += "Attempted a GET request using the URI:\n    {0}\n".format(url)
                warnings.warn(msg)
    try:
        df.to_csv(cache_path)
    except:
        if verbosity > 0 and use_cache:
            from traceback import print_exc
            print_exc()
            warnings.warn('Unable to write weather data to cache file at {}'.format(cache_path))
    return df