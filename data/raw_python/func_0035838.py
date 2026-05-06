def keyspace(self, keyspace):
        """
        Convenient, consistent access to a sub-set of all keys.
        """
        if FORMAT_SPEC.search(keyspace):
            return KeyspacedProxy(self, keyspace)
        else:
            return KeyspacedProxy(self, self._keyspaces[keyspace])