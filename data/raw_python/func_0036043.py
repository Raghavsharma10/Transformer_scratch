def list_feeds():
    """List all feeds in plain text and give their aliases"""
    with Database("feeds") as feeds, Database("aliases") as aliases_db:
        for feed in feeds:
            name = feed
            url = feeds[feed]
            aliases = []
            for k, v in zip(list(aliases_db.keys()), list(aliases_db.values())):
                if v == name:
                    aliases.append(k)
            if aliases:
                print(name, " : %s Aliases: %s" % (url, aliases))
            else:
                print(name, " : %s" % url)