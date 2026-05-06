def calculate_sunrise_sunset(locator, calc_date=datetime.utcnow()):
    """calculates the next sunset and sunrise for a Maidenhead locator at a give date & time

        Args:
            locator1 (string): Maidenhead Locator, either 4 or 6 characters
            calc_date (datetime, optional): Starting datetime for the calculations (UTC)

        Returns:
            dict: Containing datetimes for morning_dawn, sunrise, evening_dawn, sunset

        Raises:
            ValueError: When called with wrong or invalid input arg
            AttributeError: When args are not a string

        Example:
           The following calculates the next sunrise & sunset for JN48QM on the 1./Jan/2014

           >>> from pyhamtools.locator import calculate_sunrise_sunset
           >>> from datetime import datetime
           >>> import pytz
           >>> UTC = pytz.UTC
           >>> myDate = datetime(year=2014, month=1, day=1, tzinfo=UTC)
           >>> calculate_sunrise_sunset("JN48QM", myDate)
           {
               'morning_dawn': datetime.datetime(2014, 1, 1, 6, 36, 51, 710524, tzinfo=<UTC>),
               'sunset': datetime.datetime(2014, 1, 1, 16, 15, 23, 31016, tzinfo=<UTC>),
               'evening_dawn': datetime.datetime(2014, 1, 1, 15, 38, 8, 355315, tzinfo=<UTC>),
               'sunrise': datetime.datetime(2014, 1, 1, 7, 14, 6, 162063, tzinfo=<UTC>)
           }

    """
    morning_dawn = None
    sunrise = None
    evening_dawn = None
    sunset = None

    latitude, longitude = locator_to_latlong(locator)

    if type(calc_date) != datetime:
        raise ValueError

    sun = ephem.Sun()
    home = ephem.Observer()

    home.lat = str(latitude)
    home.long = str(longitude)
    home.date = calc_date

    sun.compute(home)

    try:
        nextrise = home.next_rising(sun)
        nextset = home.next_setting(sun)

        home.horizon = '-6'
        beg_twilight = home.next_rising(sun, use_center=True)
        end_twilight = home.next_setting(sun, use_center=True)

        morning_dawn = beg_twilight.datetime()
        sunrise = nextrise.datetime()

        evening_dawn = nextset.datetime()
        sunset = end_twilight.datetime()

    #if sun never sets or rises (e.g. at polar circles)
    except ephem.AlwaysUpError as e:
        morning_dawn = None
        sunrise = None
        evening_dawn = None
        sunset = None
    except ephem.NeverUpError as e:
        morning_dawn = None
        sunrise = None
        evening_dawn = None
        sunset = None

    result = {}
    result['morning_dawn'] = morning_dawn
    result['sunrise'] = sunrise
    result['evening_dawn'] = evening_dawn
    result['sunset'] = sunset

    if morning_dawn:
        result['morning_dawn'] = morning_dawn.replace(tzinfo=UTC)
    if sunrise:
        result['sunrise'] = sunrise.replace(tzinfo=UTC)
    if evening_dawn:
        result['evening_dawn'] = evening_dawn.replace(tzinfo=UTC)
    if sunset:
        result['sunset'] = sunset.replace(tzinfo=UTC)
    return result