def remoteIndexer1to2(oldIndexer):
    """
    Previously external application code was responsible for adding a
    RemoteListener to a batch work source as a reliable listener.  This
    precluded the possibility of the RemoteListener resetting itself
    unilaterally.  With version 2, RemoteListener takes control of adding
    itself as a reliable listener and keeps track of the sources with which it
    is associated.  This upgrader creates that tracking state.
    """
    newIndexer = oldIndexer.upgradeVersion(
        oldIndexer.typeName, 1, 2,
        indexCount=oldIndexer.indexCount,
        installedOn=oldIndexer.installedOn,
        indexDirectory=oldIndexer.indexDirectory)

    listeners = newIndexer.store.query(
        batch._ReliableListener,
        batch._ReliableListener.listener == newIndexer)

    for listener in listeners:
        _IndexerInputSource(
            store=newIndexer.store,
            indexer=newIndexer,
            source=listener.processor)

    return newIndexer