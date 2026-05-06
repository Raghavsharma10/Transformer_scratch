def comma_join(fields, oxford=True):
    """ Join together words. """

    def fmt(field):
        return "'%s'" % field

    if not fields:
        return "nothing"
    elif len(fields) == 1:
        return fmt(fields[0])
    elif len(fields) == 2:
        return " and ".join([fmt(f) for f in fields])
    else:
        result = ", ".join([fmt(f) for f in fields[:-1]])
        if oxford:
            result += ","
        result += " and %s" % fmt(fields[-1])
        return result