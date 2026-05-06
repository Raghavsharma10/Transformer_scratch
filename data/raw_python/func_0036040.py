def del_feed(name):
    """remove from database (and delete aliases)"""
    with Database("aliases") as aliases, Database("feeds") as feeds:
        if aliases[name]:
            proper_name = aliases[name]
        elif feeds[name]:
            proper_name = feeds[name]
        for k, v in aliases:
            if v == proper_name:
                del aliases[k]
        # deleted from aliases
        del feeds[proper_name]