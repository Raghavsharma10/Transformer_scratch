def register(self, peer):
        """
        Registers a peer according to its description

        :param peer: A Peer description bean
        :raise KeyError:
        """
        assert isinstance(peer, beans.Peer)

        with self.__lock:
            # Check presence
            peer_id = peer.peer_id
            if peer_id in self.peers:
                raise KeyError("Already known peer: {0}".format(peer))

            # Store the description
            self.peers[peer_id] = peer

            # Store in the groups
            for name in peer.groups:
                self.groups.setdefault(name, set()).add(peer_id)