def _strip(string, pattern):
    """Return complement of pattern in string"""
    m = re.compile(pattern).search(string)

    if m:
        return string[0:m.start()] + string[m.end():len(string)]
    else:
        return string