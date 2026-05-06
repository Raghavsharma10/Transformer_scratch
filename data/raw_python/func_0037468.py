def parse_python(classifiers):
    """Parse out the versions of python supported a/c classifiers."""
    prefix = 'Programming Language :: Python ::'
    python_classifiers = [c.split('::')[2].strip() for c in classifiers if c.startswith(prefix)]
    return ', '.join([c for c in python_classifiers if parse_version(c)])