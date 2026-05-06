def get_requirements(*args):
    """Get requirements from pip requirement files."""
    requirements = set()
    contents = get_contents(*args)
    for line in contents.splitlines():
        # Strip comments.
        line = re.sub(r'^#.*|\s#.*', '', line)
        # Ignore empty lines
        if line and not line.isspace():
            requirements.add(re.sub(r'\s+', '', line))
    return sorted(requirements)