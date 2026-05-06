def path_and_line(req):
    """Return the path and line number of the file from which an
    InstallRequirement came.

    """
    path, line = (re.match(r'-r (.*) \(line (\d+)\)$',
                           req.comes_from).groups())
    return path, int(line)