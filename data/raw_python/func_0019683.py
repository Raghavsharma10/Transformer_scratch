def _get_line_(filepath):
    """
    Gets each line from the file and parse the data.
    Attempt to translate the value into a python type is possible
    (falls back to string).
    """
    for line in open(filepath):
        line = line.strip()
        # allows for comments in the file
        if line.startswith('#') or '=' not in line:
            continue
        # split on the first =, allows for subsiquent `=` in strings
        key, value = line.split('=', 1)
        key = key.strip().upper()
        value = value.strip()

        if not (key and value):
            continue

        try:
            # evaluate the string before adding into environment
            # resolves any hanging (') characters
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        #return line
        yield (key, value)