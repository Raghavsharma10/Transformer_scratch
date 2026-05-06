def clean_py_files(path):
    """
    Removes all .py files.

    :param path: the path
    :return: None
    """

    for dirname, subdirlist, filelist in os.walk(path):

        for f in filelist:
            if f.endswith('py'):
                os.remove(os.path.join(dirname, f))