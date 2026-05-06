def getIndex(self, indexName, indexClass):
        """Retrieves an index with a given index name and class
        @params indexName: The index name
        @params indexClass: vertex or edge

        @return The Index object or None"""
        if indexClass == "vertex":
            try:
                return Index(indexName, indexClass, "manual",
                        self.neograph.nodes.indexes.get(indexName))
            except client.NotFoundError:
                return None
        elif indexClass == "edge":
            try:
                return Index(indexName, indexClass, "manual",
                        self.neograph.relationships.indexes.get(indexName))
            except client.NotFoundError:
                return None
        else:
            raise KeyError("Unknown Index Class (%s). Use vertex or edge"\
                    % indexClass)