def extract_version(filepath='jeni.py', name='__version__'):
    """Parse __version__ out of given Python file.

    Given jeni.py has dependencies, `from jeni import __version__` will fail.
    """
    context = {}
    for line in open(filepath):
        if name in line:
            exec(line, context)
            break
    else:
        raise RuntimeError('{} not found in {}'.format(name, filepath))
    return context[name]