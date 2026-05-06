def gdate_to_jdn(date):
    """
    Compute Julian day from Gregorian day, month and year.

    Algorithm from wikipedia's julian_day article.
    Return: The julian day number
    """
    not_jan_or_feb = (14 - date.month) // 12
    year_since_4800bc = date.year + 4800 - not_jan_or_feb
    month_since_4800bc = date.month + 12 * not_jan_or_feb - 3
    jdn = date.day + (153 * month_since_4800bc + 2) // 5 \
        + 365 * year_since_4800bc \
        + (year_since_4800bc // 4 - year_since_4800bc // 100 +
           year_since_4800bc // 400) - 32045
    return jdn