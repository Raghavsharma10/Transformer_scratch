def recursive_symlink_dirs(source_d, destination_d):
    '''
    Create dirs and symlink all files recursively from source_d, ignoring
    errors (e.g. existing files)
    '''
    func = os.symlink
    if os.name == 'nt':
        # NOTE: need to verify that default perms only allow admins to create
        # symlinks on Windows
        func = shutil.copy
    if os.path.exists(destination_d):
        os.rmdir(destination_d)
    shutil.copytree(source_d, destination_d, copy_function=func)