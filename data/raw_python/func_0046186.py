def guess_parameters(next_line):
    """
    Attempt to guess parameters based on the presence of a parenthesized
    group of identifiers.  If successful, returns a list of parameter names;
    otherwise, returns None.
    """
    match = re.search('\(([\w\s,]+)\)', next_line)
    if match:
        return [arg.strip() for arg in match.group(1).split(',')]
    else:
        return None