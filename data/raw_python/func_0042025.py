def get_link(self, peer):
        """
        Retrieves a link to the peer

        :param peer: A Peer description
        :return: A link to the peer, None if none available
        """
        assert isinstance(peer, Peer)

        for protocol in self._protocols:
            try:
                # Try to get a link
                return protocol.get_link(peer)

            except ValueError:
                # Peer can't be handled by this protocol
                pass

        # No link found
        return None