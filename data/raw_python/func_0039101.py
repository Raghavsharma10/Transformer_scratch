def cd(dest):
    """ Temporarily cd into a directory"""
    origin = os.getcwd()
    try:
        os.chdir(dest)
        yield dest
    finally:
        os.chdir(origin)