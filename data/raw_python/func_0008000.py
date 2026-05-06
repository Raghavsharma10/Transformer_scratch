def check_if_this_file_exist(filename):
    """Check if this file exist and if it's a directory

    This function will check if the given filename
    actually exists and if it's not a Directory

    Arguments:
        filename {string} -- filename

    Return:
        True  : if it's not a directory and if this file exist
        False : If it's not a file and if it's a directory
    """
    #get the absolute path
    filename = os.path.abspath(filename)

    #Boolean
    this_file_exist = os.path.exists(filename)
    a_directory = os.path.isdir(filename)

    result = this_file_exist and not a_directory
    if result == False:
        raise ValueError('The filename given was either non existent or was a directory')
    else:
        return result