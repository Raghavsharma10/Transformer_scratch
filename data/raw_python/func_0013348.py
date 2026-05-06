def getPeer(self, url):
        """
        Finds a peer by URL and return the first peer record with that URL.
        """
        peers = list(models.Peer.select().where(models.Peer.url == url))
        if len(peers) == 0:
            raise exceptions.PeerNotFoundException(url)
        return peers[0]