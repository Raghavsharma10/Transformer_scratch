def find_config_file(file_name, extra_path=None, load_user=True):
    """
    Find a configuration file in one of these directories, tried in this order:

    - A path provided as an argument
    - A path specified by the AMBRY_CONFIG environmenal variable
    - ambry in a path specified by the VIRTUAL_ENV environmental variable
    - ~/ambry
    - /etc/ambry

    :param file_name:
    :param extra_path:
    :param load_user:
    :param path:
    :return:
    """

    paths = []

    if extra_path is not None:
        paths.append(extra_path)

    if os.getenv(ENVAR.CONFIG):
        paths.append(os.getenv(ENVAR.CONFIG))

    if os.getenv(ENVAR.VIRT):
        paths.append(os.path.join(os.getenv(ENVAR.VIRT), USER_DIR))

    if load_user:
        paths.append(os.path.expanduser('~/' + USER_DIR))

    paths.append(ROOT_DIR)

    for path in paths:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, file_name)):
            f = os.path.join(path, file_name)
            return f

    raise ConfigurationError(
        "Failed to find configuration file '{}'. Looked for : {} ".format(file_name, paths))