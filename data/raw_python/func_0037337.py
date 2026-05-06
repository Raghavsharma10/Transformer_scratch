def get_readme():
    """Generate long description"""
    pandoc = None
    for path in os.environ["PATH"].split(os.pathsep):
        path = path.strip('"')
        pandoc = os.path.join(path, 'pandoc')
        if os.path.isfile(pandoc) and os.access(pandoc, os.X_OK):
            break
        else:
            pandoc = None
    try:
        if pandoc:
            cmd = [pandoc, '-t', 'rst', 'README.md']
            long_description = os.popen(' '.join(cmd)).read()
        else:
            raise ValueError
    except BaseException:
        long_description = open("README.md").read()
    return long_description