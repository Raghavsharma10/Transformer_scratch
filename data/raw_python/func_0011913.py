def deep_documents(self):
        """
        list of all documents find in subtrees of this node
        """
        tree = []
        for entry in self.contents:
            if isinstance(entry, Document):
                tree.append(entry)
            else:
                tree += entry.deep_documents
        return tree