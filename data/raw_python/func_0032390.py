def remoteIndexer2to3(oldIndexer):
    """
    The documentType keyword was added to all indexable items.  Indexes need to
    be regenerated for this to take effect.  Also, PyLucene no longer stores
    the text of messages it indexes, so deleting and re-creating the indexes
    will make them much smaller.
    """
    newIndexer = oldIndexer.upgradeVersion(
        oldIndexer.typeName, 2, 3,
        indexCount=oldIndexer.indexCount,
        installedOn=oldIndexer.installedOn,
        indexDirectory=oldIndexer.indexDirectory)
    # the 3->4 upgrader for PyLuceneIndexer calls reset(), so don't do it
    # here.  also, it won't work because it's a DummyItem
    if oldIndexer.typeName != PyLuceneIndexer.typeName:
        newIndexer.reset()
    return newIndexer