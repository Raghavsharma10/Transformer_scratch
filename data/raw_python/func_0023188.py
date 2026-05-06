def find(name):
    """Locate a filename into the shader library."""

    if op.exists(name):
        return name

    path = op.dirname(__file__) or '.'

    paths = [path] + config['include_path']

    for path in paths:
        filename = op.abspath(op.join(path, name))
        if op.exists(filename):
            return filename

        for d in os.listdir(path):
            fullpath = op.abspath(op.join(path, d))
            if op.isdir(fullpath):
                filename = op.abspath(op.join(fullpath, name))
                if op.exists(filename):
                    return filename

    return None