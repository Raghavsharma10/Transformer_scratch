def gettree(self, key, create = False):
        """
        Get a subtree node from the key (path relative to this node)
        """
        tree, _ = self._getsubitem(key + '.tmp', create)
        return tree