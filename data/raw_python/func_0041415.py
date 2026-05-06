def archive_basename(filename):
    '''returns the basename (name without extension) of a recognized archive file'''
    for archive in archive_formats:
        if filename.endswith(archive_formats[archive]['suffix']):
            return filename.rstrip('.' + archive_formats[archive]['suffix'])
    return False