def getInitialPeerList(filePath, logger=None):
    """
    Attempts to get a list of peers from a file specified in configuration.

    This file has one URL per line and can contain newlines and comments.

        # Main ga4gh node
        http://1kgenomes.ga4gh.org
        # Local intranet peer
        https://192.168.1.1

    The server will attempt to add URLs in this file to its registry at
    startup and will log a warning if the file isn't found.
    """
    ret = []
    with open(filePath) as textFile:
        ret = textFile.readlines()
    if len(ret) == 0:
        if logger:
            logger.warn("Couldn't load the initial "
                        "peer list. Try adding a "
                        "file named 'initial_peers.txt' "
                        "to {}".format(os.getcwd()))
    # Remove lines that start with a hash or are empty.
    return filter(lambda x: x != "" and not x.find("#") != -1, ret)