def findObjects(path):
    """Finds objects in pairtree.

    Given a path that corresponds to a pairtree, walk it and look for
    non-shorty (it's ya birthday) directories.
    """
    objects = []
    if not os.path.isdir(path):
        return []
    contents = os.listdir(path)
    for item in contents:
        fullPath = os.path.join(path, item)
        if not os.path.isdir(fullPath):
            # deal with a split end at this point
            # we might want to consider a normalize option
            return [path]
        else:
            if isShorty(item):
                objects = objects + findObjects(fullPath)
            else:
                objects.append(fullPath)
    return objects