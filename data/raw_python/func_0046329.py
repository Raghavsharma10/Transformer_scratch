def gather_repositories():
    """
    Collects all of the repositories. The current implementation
    searches for them in the current working directory.
    """

    for (root, dirs, files) in os.walk('.', topdown=True):
        if '.git' not in dirs:
            continue

        for dir in list(dirs):
            dirs.remove(dir)

        path = os.path.split(root)[1]
        repo = os.path.basename(path)
        yield (repo, root)