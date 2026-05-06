def _days_from_3744(hebrew_year):
    """Return: Number of days since 3,1,3744."""
    # Start point for calculation is Molad new year 3744 (16BC)
    years_from_3744 = hebrew_year - 3744
    molad_3744 = get_chalakim(1 + 6, 779)    # Molad 3744 + 6 hours in parts

    # Time in months

    # Number of leap months
    leap_months = (years_from_3744 * 7 + 1) // 19
    leap_left = (years_from_3744 * 7 + 1) % 19    # Months left of leap cycle
    months = years_from_3744 * 12 + leap_months   # Total Number of months

    # Time in parts and days
    # Molad This year + Molad 3744 - corrections
    parts = months * PARTS_IN_MONTH + molad_3744
    # 28 days in month + corrections
    days = months * 28 + parts // PARTS_IN_DAY - 2

    # Time left for round date in corrections
    # 28 % 7 = 0 so only corrections counts
    parts_left_in_week = parts % PARTS_IN_WEEK
    parts_left_in_day = parts % PARTS_IN_DAY
    week_day = parts_left_in_week // PARTS_IN_DAY

    # pylint: disable=too-many-boolean-expressions
    # pylint-comment: Splitting the 'if' below might create a bug in case
    # the order is not kept.

    # Molad ד"ר ט"ג
    if ((leap_left < 12 and week_day == 3 and
         parts_left_in_day >= get_chalakim(9 + 6, 204)) or
            # Molad ט"פקת ו"טב
            (leap_left < 7 and week_day == 2 and
             parts_left_in_day >= get_chalakim(15 + 6, 589))):
        days += 1
        week_day += 1

    # pylint: enable=too-many-boolean-expressions

    # ADU
    if week_day in (1, 4, 6):
        days += 1

    return days