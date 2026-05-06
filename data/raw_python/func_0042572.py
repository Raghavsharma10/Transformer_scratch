def create_dir(dst):
    """create directory if necessary

    :param dst: 
    """
    directory = os.path.dirname(dst)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)