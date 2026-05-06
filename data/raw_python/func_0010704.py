def locator_to_latlong (locator):
    """converts Maidenhead locator in the corresponding WGS84 coordinates

        Args:
            locator (string): Locator, either 4 or 6 characters

        Returns:
            tuple (float, float): Latitude, Longitude

        Raises:
            ValueError: When called with wrong or invalid input arg
            TypeError: When arg is not a string

        Example:
           The following example converts a Maidenhead locator into Latitude and Longitude

           >>> from pyhamtools.locator import locator_to_latlong
           >>> latitude, longitude = locator_to_latlong("JN48QM")
           >>> print latitude, longitude
           48.5208333333 9.375

        Note:
             Latitude (negative = West, positive = East)
             Longitude (negative = South, positive = North)

    """

    locator = locator.upper()

    if len(locator) == 5 or len(locator) < 4:
        raise ValueError

    if ord(locator[0]) > ord('R') or ord(locator[0]) < ord('A'):
        raise ValueError

    if ord(locator[1]) > ord('R') or ord(locator[1]) < ord('A'):
        raise ValueError

    if ord(locator[2]) > ord('9') or ord(locator[2]) < ord('0'):
        raise ValueError

    if ord(locator[3]) > ord('9') or ord(locator[3]) < ord('0'):
        raise ValueError

    if len(locator) == 6:
        if ord(locator[4]) > ord('X') or ord(locator[4]) < ord('A'):
            raise ValueError
        if ord (locator[5]) > ord('X') or ord(locator[5]) < ord('A'):
            raise ValueError

    longitude = (ord(locator[0]) - ord('A')) * 20 - 180
    latitude = (ord(locator[1]) - ord('A')) * 10 - 90
    longitude += (ord(locator[2]) - ord('0')) * 2
    latitude += (ord(locator[3]) - ord('0'))

    if len(locator) == 6:
        longitude += ((ord(locator[4])) - ord('A')) * (2 / 24)
        latitude += ((ord(locator[5])) - ord('A')) * (1 / 24)

        # move to center of subsquare
        longitude += 1 / 24
        latitude += 0.5 / 24

    else:
        # move to center of square
        longitude += 1;
        latitude += 0.5;

    return latitude, longitude