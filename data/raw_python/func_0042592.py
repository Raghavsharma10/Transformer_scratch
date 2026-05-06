def getFilePathsWithExtensionsInDirectory(dirTree, patterns, sort=True):
    """
    Returns all file paths that match any one of patterns in a
    file tree with its root at dirTree.  Sorts the paths by default.
    """
    filePaths = []
    for root, dirs, files in os.walk(dirTree):
        for filePath in files:
            for pattern in patterns:
                if fnmatch.fnmatch(filePath, pattern):
                    fullPath = os.path.join(root, filePath)
                    filePaths.append(fullPath)
                    break
    if sort:
        filePaths.sort()
    return filePaths