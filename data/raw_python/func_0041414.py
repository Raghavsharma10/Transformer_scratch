def is_archive(filename):
    '''returns boolean of whether this filename looks like an archive'''
    for archive in archive_formats:
        if filename.endswith(archive_formats[archive]['suffix']):
            return True
    return False