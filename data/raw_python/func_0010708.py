def calculate_heading_longpath(locator1, locator2):
    """calculates the heading from the first to the second locator (long path)

        Args:
            locator1 (string): Locator, either 4 or 6 characters
            locator2 (string): Locator, either 4 or 6 characters

        Returns:
            float: Long path heading in deg

        Raises:
            ValueError: When called with wrong or invalid input arg
            AttributeError: When args are not a string

        Example:
           The following calculates the long path heading from locator1 to locator2

           >>> from pyhamtools.locator import calculate_heading_longpath
           >>> calculate_heading_longpath("JN48QM", "QF67bf")
           254.3136

    """

    heading = calculate_heading(locator1, locator2)

    lp = (heading + 180)%360

    return lp