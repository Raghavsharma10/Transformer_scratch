def check_existance(f):
    """Check if the file supplied as input exists."""
    if not opath.isfile(f):
        logging.error("Nanoget: File provided doesn't exist or the path is incorrect: {}".format(f))
        sys.exit("File provided doesn't exist or the path is incorrect: {}".format(f))