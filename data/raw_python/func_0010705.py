def calculate_distance(locator1, locator2):
    """calculates the (shortpath) distance between two Maidenhead locators

        Args:
            locator1 (string): Locator, either 4 or 6 characters
            locator2 (string): Locator, either 4 or 6 characters

        Returns:
            float: Distance in km

        Raises:
            ValueError: When called with wrong or invalid input arg
            AttributeError: When args are not a string

        Example:
           The following calculates the distance between two Maidenhead locators in km

           >>> from pyhamtools.locator import calculate_distance
           >>> calculate_distance("JN48QM", "QF67bf")
           16466.413

    """

    R = 6371 #earh radius
    lat1, long1 = locator_to_latlong(locator1)
    lat2, long2 = locator_to_latlong(locator2)

    d_lat = radians(lat2) - radians(lat1)
    d_long = radians(long2) - radians(long1)

    r_lat1 = radians(lat1)
    r_long1 = radians(long1)
    r_lat2 = radians(lat2)
    r_long2 = radians(long2)

    a = sin(d_lat/2) * sin(d_lat/2) + cos(r_lat1) * cos(r_lat2) * sin(d_long/2) * sin(d_long/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c #distance in km

    return d;