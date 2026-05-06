def make_dirs_if_dont_exist(path):
    """ Create directories in path if they do not exist """
    if path[-1] not in ['/']: path += '/'
    path = os.path.dirname(path)
    if path != '':
        try: os.makedirs(path)
        except OSError: pass