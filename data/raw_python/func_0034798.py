def _addTags(tags, objects):
    """ Adds tags to objects """
    for t in tags:
        for o in objects:
            o.tags.add(t)

    return True