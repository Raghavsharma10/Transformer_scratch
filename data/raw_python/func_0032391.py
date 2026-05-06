def pyLuceneIndexer4to5(old):
    """
    Copy attributes, reset index due because information about deleted
    documents has been lost, and power up for IFulltextIndexer so other code
    can find this item.
    """
    new = old.upgradeVersion(PyLuceneIndexer.typeName, 4, 5,
                             indexCount=old.indexCount,
                             installedOn=old.installedOn,
                             indexDirectory=old.indexDirectory)
    new.reset()
    new.store.powerUp(new, ixmantissa.IFulltextIndexer)
    return new