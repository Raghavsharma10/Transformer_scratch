def generate_key(filepath):
    ''' generates a new, random secret key at the given location on the
    filesystem and returns its path
    '''
    fs = path.abspath(path.expanduser(filepath))
    with open(fs, 'wb') as outfile:
        outfile.write(Fernet.generate_key())
    chmod(fs, 0o400)
    return fs