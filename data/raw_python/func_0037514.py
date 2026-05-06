def type_converter(text):
    """ I convert strings into integers, floats, and strings! """
    if text.isdigit():
        return int(text), int

    try:
        return float(text), float
    except ValueError:
        return text, STRING_TYPE