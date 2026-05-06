def get(self, key, value):
        """Gets an element from an index under a given
        key-value pair
        @params key: Index key string
        @params value: Index value string
        @returns A generator of Vertex or Edge objects"""
        for element in self.neoindex[key][value]:
            if self.indexClass == "vertex":
                yield Vertex(element)
            elif self.indexClass == "edge":
                yield Edge(element)
            else:
                raise TypeError(self.indexClass)