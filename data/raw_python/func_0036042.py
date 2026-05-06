def alias_feed(name, alias):
    """write aliases to db"""
    with Database("aliases") as db:
        if alias in db:
            print("Something has gone horribly wrong with your aliases! Try deleting the %s entry." % name)
            return
        else:
            db[alias] = name