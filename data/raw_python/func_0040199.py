def read_file(fname):
    """
    Read file, convert wildcards into regular expressions, skip empty lines
    and comments.
    """
    res = []
    try:
        with open(fname, 'r') as f:
            for line in f:
                line = line.rstrip('\n').rstrip('\r')
                if line and (line[0] != '#'):
                    regexline = ".*" + re.sub("\*", ".*", line) + ".*"
                    res.append(regexline.lower())
    except IOError:
        pass
    return res