def date_struct_nn(year, month, day, tz="UTC"):
    """
    Assemble a date object but if day or month is none set them to 1
    to make it easier to deal with partial dates
    """
    if not day:
        day = 1
    if not month:
        month = 1
    return date_struct(year, month, day, tz)