def finddirs(root):
    """Return a list of all the directories under `root`"""
    retval = []
    for root, dirs, files in os.walk(root):
        for d in dirs:
            retval.append(os.path.join(root, d))
    return retval