def send(self, peer_id, message):
        """
        Synchronously sends a message

        :param peer_id: UUID of a peer
        :param message: Message to send to the peer
        :raise KeyError: Unknown peer
        :raise ValueError: No link to the peer
        """
        assert isinstance(message, beans.RawMessage)

        # Get peer description (raises KeyError)
        peer = self._directory.get_peer(peer_id)

        # Get a link to the peer (raises ValueError)
        link = self._get_link(peer)
        assert isinstance(link, beans.AbstractLink)

        # Call the link, and return its result
        return link.send(message)