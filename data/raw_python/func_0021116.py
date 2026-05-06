def get_entity(package, entity):
    """
    eg, get_entity('spectator', 'version') returns `__version__` value in
    `__init__.py`.
    """
    init_py = open(os.path.join(package, '__init__.py')).read()
    find = "__%s__ = ['\"]([^'\"]+)['\"]" % entity
    return re.search(find, init_py).group(1)