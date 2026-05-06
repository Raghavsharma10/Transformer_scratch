def get_pycons3rt_home_dir():
    """Returns the pycons3rt home directory based on OS

    :return: (str) Full path to pycons3rt home
    :raises: OSError
    """
    if platform.system() == 'Linux':
        return os.path.join(os.path.sep, 'etc', 'pycons3rt')
    elif platform.system() == 'Windows':
        return os.path.join('C:', os.path.sep, 'pycons3rt')
    elif platform.system() == 'Darwin':
        return os.path.join(os.path.expanduser('~'), '.pycons3rt')
    else:
        raise OSError('Unsupported Operating System')