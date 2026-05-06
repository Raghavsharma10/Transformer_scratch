def replace_all(filepath, searchExp, replaceExp):
    """
    Replace all the ocurrences (in a file) of a string with another value.
    """
    for line in fileinput.input(filepath, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)