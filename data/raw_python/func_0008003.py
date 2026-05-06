def get_json(filename):
    """ Return a json value of the exif

    Get a filename and return a JSON object

    Arguments:
        filename {string} -- your filename

    Returns:
        [JSON] -- Return a JSON object
    """
    check_if_this_file_exist(filename)

    #Process this function
    filename = os.path.abspath(filename)
    s = command_line(['exiftool', '-G', '-j', '-sort', filename])
    if s:
        #convert bytes to string
        s = s.decode('utf-8').rstrip('\r\n')
        return json.loads(s)
    else:
        return s