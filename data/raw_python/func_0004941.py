def parse_number(val, use_dateutilparser=False):
    """Try to auto-detect the numeric type of the value. First a conversion to
    int is tried. If this fails float is tried, and if that fails too, unicode()
    is executed. If this also fails, a ValueError is raised.
    """
    if use_dateutilparser:
        funcs = [int, float, parse_list_from_string,
                 dateutil.parser.parse, str]
    else:
        funcs = [int, float, parse_list_from_string, str]
    if (val.strip().startswith("'") and val.strip().endswith("'")) or (val.strip().startswith('"') and val.strip().endswith('"')):
        return val[1:-1]
    for f in funcs:
        try:
            return f(val)
        # eat exception
        except (ValueError, UnicodeEncodeError, UnicodeDecodeError) as ve:
            pass
    raise ValueError('Cannot parse number:', val)