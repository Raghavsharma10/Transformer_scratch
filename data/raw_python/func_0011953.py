def peers(self):
        "list of the (host, port) pairs of all connected peer Hubs"
        return [addr for (addr, peer) in self._dispatcher.peers.items()
                if peer.up]