def parse_django(classifiers):
    """Parse out the versions of django supported a/c classifiers."""
    prefix = 'Framework :: Django ::'
    django_classifiers = [c.split('::')[2].strip() for c in classifiers if c.startswith(prefix)]
    return ', '.join([c for c in django_classifiers if parse_version(c)])