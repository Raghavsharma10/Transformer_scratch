def find_file_dir_up(filename, path=None, n=None):
    '''Finding file in directory upwards.
    '''
    if path is None:
        path = os.getcwd()
    i = 0
    while True:
        current_path = path
        for _ in range(i):
            current_path = os.path.split(current_path)[0]
        if os.path.isfile(os.path.join(current_path, filename)):  # found file and return
            return os.path.join(current_path, filename)
        elif os.path.dirname(current_path) == current_path:  # root of filesystem
            return
        elif n is not None and i == n:
            return
        else:  # file not found
            i += 1
            continue