def get_file_list(path):
    """ Recursively lists all files in a file system below 'path'. """
    f_list = []
    def recur_dir(path, newpath = os.path.sep):
        files = os.listdir(path)
        for fle in files:
            f_path = cpjoin(path, fle)
            if os.path.isdir(f_path): recur_dir(f_path, cpjoin(newpath, fle))
            elif os.path.isfile(f_path): f_list.append(get_single_file_info(f_path, cpjoin(newpath, fle)))

    recur_dir(path)
    return f_list