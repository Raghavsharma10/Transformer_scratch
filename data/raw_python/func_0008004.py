def get_csv(filename):
    """ Return a csv representation of the exif

    get a filename and returns a unicode string with a CSV format

    Arguments:
        filename {string} -- your filename

    Returns:
        [unicode] -- unicode string
    """
    check_if_this_file_exist(filename)

    #Process this function
    filename = os.path.abspath(filename)
    s = command_line(['exiftool', '-G', '-csv', '-sort', filename])
    if s:
        #convert bytes to string
        s = s.decode('utf-8')
        return s
    else:
        return 0