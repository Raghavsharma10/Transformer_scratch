def matching_line(lines, keyword):
    """ Returns the first matching line in a list of lines.
    @see match()
    """
    for line in lines:
        matching = match(line,keyword)
        if matching != None:
            return matching
    return None