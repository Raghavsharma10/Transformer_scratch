def temp_path(file_name=None):
    """
    Gets a temp path.

    Kwargs:
        file_name (str) : if file name is specified, it gets appended to the temp dir.

    Usage::

        temp_file_path = temp_path("myfile")
        copyfile("myfile", temp_file_path) # copies 'myfile' to '/tmp/myfile'

    """

    if file_name is None:
        file_name = generate_timestamped_string("wtf_temp_file")

    return os.path.join(tempfile.gettempdir(), file_name)