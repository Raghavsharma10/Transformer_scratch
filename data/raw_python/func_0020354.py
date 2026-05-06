def usage():
    """Print out a usage message"""

    global options
    l = len(options['long'])
    options['shortlist'] = [s for s in options['short'] if s is not ":"]

    print("python -m behave2cucumber [-h] [-d level|--debug=level]")
    for i in range(l):
        print("    -{0}|--{1:20} {2}".format(options['shortlist'][i], options['long'][i], options['descriptions'][i]))