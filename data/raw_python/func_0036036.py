def resolve_name(name):
    """Takes a given input from a user and finds the url for it"""
    logger.debug("resolve_name: %s", name)
    with Database("feeds") as feeds, Database("aliases") as aliases:
        if name in aliases.keys():
            return feeds[aliases[name]]
        elif name in feeds.keys():
            return feeds[name]
        else:
            print("Cannot find feed named: %s" % name)
            return