def get_packaged_files(package_name):
    """ Collect relative paths to all files which have already been packaged """
    if not os.path.isdir('dist'):
        return []
    return [os.path.join('dist', filename) for filename in os.listdir('dist')]