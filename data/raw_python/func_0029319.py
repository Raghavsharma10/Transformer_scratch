def get_version(path="src/devpy/__init__.py"):
    """ Return the version of by with regex intead of importing it"""
    init_content = open(path, "rt").read()
    pattern = r"^__version__ = ['\"]([^'\"]*)['\"]"
    return re.search(pattern, init_content, re.M).group(1)