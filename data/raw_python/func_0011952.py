def add_peer(self, peer_addr):
        "Build a connection to the Hub at a given ``(host, port)`` address"
        peer = connection.Peer(
                self._ident, self._dispatcher, peer_addr, backend.Socket())
        peer.start()
        self._started_peers[peer_addr] = peer