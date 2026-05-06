def hdate_to_jdn(date):
    """
    Compute Julian day from Hebrew day, month and year.

    Return: julian day number,
            1 of tishrey julians,
            1 of tishrey julians next year
    """
    day = date.day
    month = date.month
    if date.month == 13:
        month = 6
    if date.month == 14:
        month = 6
        day += 30

    # Calculate days since 1,1,3744
    day = _days_from_3744(date.year) + (59 * (month - 1) + 1) // 2 + day

    # length of year
    length_of_year = get_size_of_hebrew_year(date.year)
    # Special cases for this year
    if length_of_year % 10 > 4 and month > 2:  # long Heshvan
        day += 1
    if length_of_year % 10 < 4 and month > 3:  # short Kislev
        day -= 1
    if length_of_year > 365 and month > 6:  # leap year
        day += 30

    # adjust to julian
    return day + 1715118