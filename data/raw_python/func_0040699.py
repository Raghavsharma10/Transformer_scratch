def unregister(self, peer_id):
        """
        Unregisters the given peer

        :param peer_id: A peer UUID
        :raise KeyError: Unknown peer
        """
        with self.__lock:
            # Pop it from accesses (will raise a KeyError if absent)
            peer = self.peers.pop(peer_id)
            assert isinstance(peer, beans.Peer)

            # Remove it from groups
            for name in peer.groups:
                try:
                    # Clean up the group
                    group = self.groups[name]
                    group.remove(peer_id)

                    # Remove it if it's empty
                    if not group:
                        del self.groups[name]

                except KeyError:
                    # Be tolerant here
                    pass