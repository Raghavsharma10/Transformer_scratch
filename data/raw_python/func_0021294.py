def match(line, keyword):
    """ If the first part of line (modulo blanks) matches keyword,
    returns the end of that line. Otherwise checks if keyword is
    anywhere in the line and returns that section, else returns None"""

    line = line.lstrip()
    length = len(keyword)
    if line[:length] == keyword:
        return line[length:]
    else:
        if keyword in line:
            return line[line.index(keyword):]
        else:
            return None