def latlong_to_locator (latitude, longitude):
    """converts WGS84 coordinates into the corresponding Maidenhead Locator

        Args:
            latitude (float): Latitude
            longitude (float): Longitude

        Returns:
            string: Maidenhead locator

        Raises:
            ValueError: When called with wrong or invalid input args
            TypeError: When args are non float values

        Example:
           The following example converts latitude and longitude into the Maidenhead locator

           >>> from pyhamtools.locator import latlong_to_locator
           >>> latitude = 48.5208333
           >>> longitude = 9.375
           >>> latlong_to_locator(latitude, longitude)
           'JN48QM'

        Note:
             Latitude (negative = West, positive = East)
             Longitude (negative = South, positive = North)

    """

    if longitude >= 180 or longitude <= -180:
        raise ValueError

    if latitude >= 90 or latitude <= -90:
        raise ValueError

    longitude += 180;
    latitude +=90;

    locator = chr(ord('A') + int(longitude / 20))
    locator += chr(ord('A') + int(latitude / 10))
    locator += chr(ord('0') + int((longitude % 20) / 2))
    locator += chr(ord('0') + int(latitude % 10))
    locator += chr(ord('A') + int((longitude - int(longitude / 2) * 2) / (2 / 24)))
    locator += chr(ord('A') + int((latitude - int(latitude / 1) * 1 ) / (1 / 24)))

    return locator