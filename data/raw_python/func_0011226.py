def list_projects(root, backend=os.listdir):
    """List projects at `root`

    Arguments:
        root (str): Absolute path to the `be` root directory,
            typically the current working directory.

    """
    projects = list()
    for project in sorted(backend(root)):
        abspath = os.path.join(root, project)
        if not isproject(abspath):
            continue
        projects.append(project)
    return projects