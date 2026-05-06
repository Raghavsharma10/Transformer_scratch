def remove(self, key, value, element):
        """Removes an element from an index under a given
        key-value pair
        @params key: Index key string
        @params value: Index value string
        @params element: Vertex or Edge element to be removed"""
        self.neoindex.delete(key, value, element.neoelement)