def add(self, document):
        """
        Add a document to the database.
        """
        docid = int(document.uniqueIdentifier())
        text = u' '.join(document.textParts())

        self.store.executeSQL(self.addSQL, (docid, text))