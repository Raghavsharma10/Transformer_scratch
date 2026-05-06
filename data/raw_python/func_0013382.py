def removePeer(self, url):
        """
        Remove peers by URL.
        """
        q = models.Peer.delete().where(
            models.Peer.url == url)
        q.execute()