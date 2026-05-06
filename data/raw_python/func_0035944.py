def find_build_dir(path, build="_build"):
    """try to guess the build folder's location"""
    path = os.path.abspath(os.path.expanduser(path))
    contents = os.listdir(path)
    filtered_contents = [directory for directory in contents
                            if os.path.isdir(os.path.join(path, directory))]

    if build in filtered_contents:
        return os.path.join(path, build)
    else:
        if path == os.path.realpath("/"):
            return None
        else:
            return find_build_dir("{0}/..".format(path), build)