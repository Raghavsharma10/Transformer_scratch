def _safe_cast(out_type, val):
    """Try to covert val to out_type but never raise an exception. If
    the value can't be converted, then a sensible default value is
    returned. out_type should be bool, int, or unicode; otherwise, the
    value is just passed through.
    """
    if val is None:
        return None

    if out_type == int:
        if isinstance(val, int) or isinstance(val, float):
            # Just a number.
            return int(val)
        else:
            # Process any other type as a string.
            if isinstance(val, bytes):
                val = val.decode('utf-8', 'ignore')
            elif not isinstance(val, six.string_types):
                val = six.text_type(val)
            # Get a number from the front of the string.
            match = re.match(r'[\+-]?[0-9]+', val.strip())
            return int(match.group(0)) if match else 0

    elif out_type == bool:
        try:
            # Should work for strings, bools, ints:
            return bool(int(val))
        except ValueError:
            return False

    elif out_type == six.text_type:
        if isinstance(val, bytes):
            return val.decode('utf-8', 'ignore')
        elif isinstance(val, six.text_type):
            return val
        else:
            return six.text_type(val)

    elif out_type == float:
        if isinstance(val, int) or isinstance(val, float):
            return float(val)
        else:
            if isinstance(val, bytes):
                val = val.decode('utf-8', 'ignore')
            else:
                val = six.text_type(val)
            match = re.match(r'[\+-]?([0-9]+\.?[0-9]*|[0-9]*\.[0-9]+)',
                             val.strip())
            if match:
                val = match.group(0)
                if val:
                    return float(val)
            return 0.0

    else:
        return val