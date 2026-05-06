def jdn_to_gdate(jdn):
    """
    Convert from the Julian day to the Gregorian day.

    Algorithm from 'Julian and Gregorian Day Numbers' by Peter Meyer.
    Return: day, month, year
    """
    # pylint: disable=invalid-name

    # The algorithm is a verbatim copy from Peter Meyer's article
    # No explanation in the article is given for the variables
    # Hence the exceptions for pylint and for flake8 (E741)

    l = jdn + 68569  # noqa: E741
    n = (4 * l) // 146097
    l = l - (146097 * n + 3) // 4  # noqa: E741
    i = (4000 * (l + 1)) // 1461001  # that's 1,461,001
    l = l - (1461 * i) // 4 + 31  # noqa: E741
    j = (80 * l) // 2447
    day = l - (2447 * j) // 80
    l = j // 11  # noqa: E741
    month = j + 2 - (12 * l)
    year = 100 * (n - 49) + i + l  # that's a lower-case L

    return datetime.date(year, month, day)