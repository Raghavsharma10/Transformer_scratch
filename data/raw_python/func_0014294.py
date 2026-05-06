def extant_file(path):
    """
    Check if file exists with argparse
    """
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("{} does not exist".format(path))
    return path