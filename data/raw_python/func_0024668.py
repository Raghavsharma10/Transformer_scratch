def parse_string(s):
    """Parse a string with units and try to make a bitmath object out of
it.

String inputs may include whitespace characters between the value and
the unit.
    """
    # Strings only please
    if not isinstance(s, (str, unicode)):
        raise ValueError("parse_string only accepts string inputs but a %s was given" %
                         type(s))

    # get the index of the first alphabetic character
    try:
        index = list([i.isalpha() for i in s]).index(True)
    except ValueError:
        # If there's no alphabetic characters we won't be able to .index(True)
        raise ValueError("No unit detected, can not parse string '%s' into a bitmath object" % s)

    # split the string into the value and the unit
    val, unit = s[:index], s[index:]

    # see if the unit exists as a type in our namespace

    if unit == "b":
        unit_class = Bit
    elif unit == "B":
        unit_class = Byte
    else:
        if not (hasattr(sys.modules[__name__], unit) and isinstance(getattr(sys.modules[__name__], unit), type)):
            raise ValueError("The unit %s is not a valid bitmath unit" % unit)
        unit_class = globals()[unit]

    try:
        val = float(val)
    except ValueError:
        raise
    try:
        return unit_class(val)
    except:  # pragma: no cover
        raise ValueError("Can't parse string %s into a bitmath object" % s)