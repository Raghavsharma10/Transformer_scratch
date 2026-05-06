def jdn_to_hdate(jdn):
    """Convert from the Julian day to the Hebrew day."""
    # calculate Gregorian date
    date = jdn_to_gdate(jdn)

    # Guess Hebrew year is Gregorian year + 3760
    year = date.year + 3760

    jdn_tishrey1 = hdate_to_jdn(HebrewDate(year, 1, 1))
    jdn_tishrey1_next_year = hdate_to_jdn(HebrewDate(year + 1, 1, 1))

    # Check if computed year was underestimated
    if jdn_tishrey1_next_year <= jdn:
        year = year + 1
        jdn_tishrey1 = jdn_tishrey1_next_year
        jdn_tishrey1_next_year = hdate_to_jdn(HebrewDate(year + 1, 1, 1))

    size_of_year = get_size_of_hebrew_year(year)

    # days into this year, first month 0..29
    days = jdn - jdn_tishrey1

    # last 8 months always have 236 days
    if days >= (size_of_year - 236):  # in last 8 months
        days = days - (size_of_year - 236)
        month = days * 2 // 59
        day = days - (month * 59 + 1) // 2 + 1

        month = month + 4 + 1

        # if leap
        if size_of_year > 355 and month <= 6:
            month = month + 8
    else:  # in 4-5 first months
        # Special cases for this year
        if size_of_year % 10 > 4 and days == 59:   # long Heshvan (day 30)
            month = 1
            day = 30
        elif size_of_year % 10 > 4 and days > 59:  # long Heshvan
            month = (days - 1) * 2 // 59
            day = days - (month * 59 + 1) // 2
        elif size_of_year % 10 < 4 and days > 87:  # short kislev
            month = (days + 1) * 2 // 59
            day = days - (month * 59 + 1) // 2 + 2
        else:  # regular months
            month = days * 2 // 59
            day = days - (month * 59 + 1) // 2 + 1

        month = month + 1

    return HebrewDate(year, month, day)