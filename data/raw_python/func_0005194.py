def remove_blank_lines(string):
    """ Removes all blank lines in @string

        -> #str without blank lines
    """
    return "\n".join(line
                     for line in string.split("\n")
                     if len(line.strip()))