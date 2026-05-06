def createManualIndex(self, indexName, indexClass):
        """Creates an index manually managed
        @params name: The index name
        @params indexClass: vertex or edge

        @returns The created Index"""
        indexClass = str(indexClass).lower()
        if indexClass == "vertex":
            index = self.neograph.nodes.indexes.create(indexName)
        elif indexClass == "edge":
            index = self.neograph.relationships.indexes.create(indexName)
        else:
            NameError("Unknown Index Class %s" % indexClass)
        return Index(indexName, indexClass, "manual", index)