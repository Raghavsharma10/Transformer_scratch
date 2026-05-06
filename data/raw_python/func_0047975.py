def out(*args):
    """ Outputs its parameters to users stdout. """
    for value in args:
        sys.stdout.write(value)

    sys.stdout.write(os.linesep)