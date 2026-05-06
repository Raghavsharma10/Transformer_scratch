def getPeer(self, url):
        """
        Select the first peer in the datarepo with the given url simulating
        the behavior of selecting by URL. This is only used during testing.
        """
        peers = filter(lambda x: x.getUrl() == url, self.getPeers())
        if len(peers) == 0:
            raise exceptions.PeerNotFoundException(url)
        return peers[0]