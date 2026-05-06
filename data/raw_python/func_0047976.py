def err(*args):
    """ Outputs its parameters to users stderr. """
    for value in args:
        sys.stderr.write(value)

    sys.stderr.write(os.linesep)