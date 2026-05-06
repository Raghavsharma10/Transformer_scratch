def mkdir(dir, enter):
    """Create directory with template for topic of the current environment

    """

    if not os.path.exists(dir):
        os.makedirs(dir)