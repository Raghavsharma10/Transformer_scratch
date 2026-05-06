def _get_link(self, peer):
        """
        Returns a link to the given peer

        :return: A Link object
        :raise ValueError: Unknown peer
        """
        assert isinstance(peer, beans.Peer)

        # Look for a link to the peer, using routers
        for router in self._routers:
            link = router.get_link(peer)
            if link:
                return link

        # Not found
        raise ValueError("No link to peer {0}".format(peer))