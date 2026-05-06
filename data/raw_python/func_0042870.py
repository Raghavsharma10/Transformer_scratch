def createArgumentParser(description):
    """
    Create an argument parser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=SortedHelpFormatter)
    return parser