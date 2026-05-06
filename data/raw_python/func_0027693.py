def createSomeItems(store, itemType, values, counter):
    """
    Create some instances of a particular type in a store.
    """
    for i in counter:
        itemType(store=store, **values)