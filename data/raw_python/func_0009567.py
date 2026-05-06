def read(*paths):
    """Build a file path from *paths* and return the contents."""
    try:
        f_name = os.path.join(*paths)
        with open(f_name, 'r') as f:
            return f.read()
    except IOError:
        print('%s not existing ... skipping' % f_name)
        return ''