def get_single_file_info(f_path, int_path):
    """ Gets the creates and last change times for a single file,
    f_path is the path to the file on disk, int_path is an internal
    path relative to a root directory.  """
    return { 'path'     : force_unicode(int_path),
             'created'  : os.path.getctime(f_path),
             'last_mod' : os.path.getmtime(f_path)}