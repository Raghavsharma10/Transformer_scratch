def initialize_pycons3rt_dirs():
    """Initializes the pycons3rt directories

    :return: None
    :raises: OSError
    """
    for pycons3rt_dir in [get_pycons3rt_home_dir(),
                          get_pycons3rt_user_dir(),
                          get_pycons3rt_conf_dir(),
                          get_pycons3rt_log_dir(),
                          get_pycons3rt_src_dir()]:
        if os.path.isdir(pycons3rt_dir):
            continue
        try:
            os.makedirs(pycons3rt_dir)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(pycons3rt_dir):
                pass
            else:
                msg = 'Unable to create directory: {d}'.format(d=pycons3rt_dir)
                raise OSError(msg)