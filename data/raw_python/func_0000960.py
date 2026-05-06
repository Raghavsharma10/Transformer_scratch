def references_date(year=None):
    "Handle year value parsing for some edge cases"
    date = None
    discriminator = None
    in_press = None
    if year and "in press" in year.lower().strip():
        in_press = True
    elif year and re.match("^[0-9]+$", year):
        date = year
    elif year:
        discriminator_match = re.match("^([0-9]+?)([a-z]+?)$", year)
        if discriminator_match:
            date = discriminator_match.group(1)
            discriminator = discriminator_match.group(2)
        else:
            date = year
    return (date, discriminator, in_press)