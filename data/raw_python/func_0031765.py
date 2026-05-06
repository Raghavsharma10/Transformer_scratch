def tag(name, tag_name):
    """
    Tag the named metric with the given tag.
    """

    with LOCK:
        # just to check if <name> exists
        metric(name)

        TAGS.setdefault(tag_name, set()).add(name)