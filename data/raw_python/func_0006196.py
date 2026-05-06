def get_node(self, key):
        """Return node for a given key. Else return None."""
        pos = self._get_node_pos(key)
        if pos is None:
            return None
        return self._hashring[self._sorted_keys[pos]]