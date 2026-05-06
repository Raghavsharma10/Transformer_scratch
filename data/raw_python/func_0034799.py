def _removeTags(tags, objects):
    """ Removes tags from objects """
    for t in tags:
        for o in objects:
            o.tags.remove(t)

    return True