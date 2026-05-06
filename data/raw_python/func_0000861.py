def first_parent(tag, nodename):
    """
    Given a beautiful soup tag, look at its parents and return the first
    tag name that matches nodename or the list nodename
    """
    if nodename is not None and type(nodename) == str:
        nodename = [nodename]
    return first(list(filter(lambda tag: tag.name in nodename, tag.parents)))