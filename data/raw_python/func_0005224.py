def within_proj_dir(path='.'):
    # type: (Optional[str]) -> str
    """ Return an absolute path to the given project relative path.

    :param path:
        Project relative path that will be converted to the system wide absolute
        path.
    :return:
        Absolute path.
    """
    curr_dir = os.getcwd()

    os.chdir(proj_path(path))

    yield

    os.chdir(curr_dir)