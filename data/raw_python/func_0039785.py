def put(self, key, value, element):
        """Puts an element in an index under a given
        key-value pair
        @params key: Index key string
        @params value: Index value string
        @params element: Vertex or Edge element to be indexed"""
        self.neoindex[key][value] = element.neoelement