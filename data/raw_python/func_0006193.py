def _get_node_pos(self, key):
        """Return node position(integer) for a given key or None."""
        if not self._hashring:
            return

        k = md5_bytes(key)
        key = (k[3] << 24) | (k[2] << 16) | (k[1] << 8) | k[0]

        nodes = self._sorted_keys
        pos = bisect(nodes, key)

        if pos == len(nodes):
            return 0
        return pos