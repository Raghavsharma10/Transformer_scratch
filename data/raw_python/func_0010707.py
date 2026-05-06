def calculate_heading(locator1, locator2):
    """calculates the heading from the first to the second locator

        Args:
            locator1 (string): Locator, either 4 or 6 characters
            locator2 (string): Locator, either 4 or 6 characters

        Returns:
            float: Heading in deg

        Raises:
            ValueError: When called with wrong or invalid input arg
            AttributeError: When args are not a string

        Example:
           The following calculates the heading from locator1 to locator2

           >>> from pyhamtools.locator import calculate_heading
           >>> calculate_heading("JN48QM", "QF67bf")
           74.3136

    """

    lat1, long1 = locator_to_latlong(locator1)
    lat2, long2 = locator_to_latlong(locator2)

    r_lat1 = radians(lat1)
    r_lon1 = radians(long1)

    r_lat2 = radians(lat2)
    r_lon2 = radians(long2)

    d_lon = radians(long2 - long1)

    b = atan2(sin(d_lon)*cos(r_lat2),cos(r_lat1)*sin(r_lat2)-sin(r_lat1)*cos(r_lat2)*cos(d_lon)) # bearing calc
    bd = degrees(b)
    br,bn = divmod(bd+360,360) # the bearing remainder and final bearing

    return bn