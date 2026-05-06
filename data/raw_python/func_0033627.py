def recursive_hardlink_dirs(source_d, destination_d):
    '''
    Same as above, except creating hardlinks for all files
    '''
    func = os.link
    if os.name == 'nt':
        func = shutil.copy
    if os.path.exists(destination_d):
        os.rmdir(destination_d)
    shutil.copytree(source_d, destination_d, copy_function=func)