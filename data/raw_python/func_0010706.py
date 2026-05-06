def calculate_distance_longpath(locator1, locator2):
    """calculates the (longpath) distance between two Maidenhead locators

        Args:
            locator1 (string): Locator, either 4 or 6 characters
            locator2 (string): Locator, either 4 or 6 characters

        Returns:
            float: Distance in km

        Raises:
            ValueError: When called with wrong or invalid input arg
            AttributeError: When args are not a string

        Example:
           The following calculates the longpath distance between two Maidenhead locators in km

           >>> from pyhamtools.locator import calculate_distance_longpath
           >>> calculate_distance_longpath("JN48QM", "QF67bf")
           23541.5867

    """

    c = 40008 #[km] earth circumference
    sp = calculate_distance(locator1, locator2)

    return c - sp