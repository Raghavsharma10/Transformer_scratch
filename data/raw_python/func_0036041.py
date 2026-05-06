def del_alias(alias):
    """sometimes you goof up."""
    with Database("aliases") as mydb:
        try:
            print("removing alias of %s to %s" % (alias, mydb.pop(alias)))
        except KeyError:
            print("No such alias key")
            print("Check alias db:")
            print(zip(list(mydb.keys()), list(mydb.values())))