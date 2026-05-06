def save_file(path, text):
    """
    Save a string to a file.  If the containing directory(ies) doesn't exist,
    this creates it.
    """
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    fd = open(path, 'w')
    try:
        fd.write(text)
    finally:
        fd.close()