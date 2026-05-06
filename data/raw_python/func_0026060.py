def from_jd(jdc):
    "Create a new date from a Julian date."
    cdc = floor(jdc) + 0.5 - EPOCH
    year = floor((cdc - floor((cdc + 366) / 1461)) / 365) + 1

    yday = jdc - to_jd(year, 1, 1)

    month = floor(yday / 30) + 1
    day = yday - (month - 1) * 30 + 1
    return year, month, day