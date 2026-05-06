def has_external_dependency(name):
    'Check that a non-Python dependency is installed.'
    for directory in os.environ['PATH'].split(':'):
        if os.path.exists(os.path.join(directory, name)):
            return True
    return False