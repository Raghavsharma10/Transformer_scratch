def to_jd(year, month, day):
    "Retrieve the Julian date equivalent for this date"
    return day + (month - 1) * 30 + (year - 1) * 365 + floor(year / 4) + EPOCH - 1