def getPeers(self, offset=0, limit=1000):
        """
        Get the list of peers using an SQL offset and limit. Returns a list
        of peer datamodel objects in a list.
        """
        select = models.Peer.select().order_by(
            models.Peer.url).limit(limit).offset(offset)
        return [peers.Peer(p.url, record=p) for p in select]