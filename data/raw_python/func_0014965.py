def get_lib_filename(category, name):
    """ Get a filename of a built-in library file. """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if category == 'js':
        filename = os.path.join('js', '{0}.js'.format(name))
    elif category == 'css':
        filename = os.path.join('css', '{0}.css'.format(name))
    elif category == 'html':
        filename = os.path.join('html', '{0}.html'.format(name))
    else:
        raise ValueError("Unknown category")
    return os.path.join(base_dir, 'lib', filename)