def insertInitialPeer(dataRepository, url, logger=None):
    """
    Takes the datarepository, a url, and an optional logger and attempts
    to add the peer into the repository.
    """
    insertPeer = dataRepository.insertPeer
    try:
        peer = datamodel.peers.Peer(url)
        insertPeer(peer)
    except exceptions.RepoManagerException as exc:
        if logger:
            logger.debug(
                "Peer already in registry {} {}".format(peer.getUrl(), exc))
    except exceptions.BadUrlException as exc:
        if logger:
            logger.debug("A URL in the initial "
                         "peer list {} was malformed. {}".format(url), exc)