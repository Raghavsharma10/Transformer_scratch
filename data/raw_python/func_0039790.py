def getIndices(self):
        """Returns a generator function over all the existing indexes

        @returns A generator function over all rhe Index objects"""
        for indexName in self.neograph.nodes.indexes.keys():
            indexObject = self.neograph.nodes.indexes.get(indexName)
            yield Index(indexName, "vertex", "manual", indexObject)
        for indexName in self.neograph.relationships.indexes.keys():
            indexObject = self.neograph.relationships.indexes.get(indexName)
            yield Index(indexName, "edge", "manual", indexObject)