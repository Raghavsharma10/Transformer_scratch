def remove(self, docid):
        """
        Remove a document from the database.
        """
        docid = int(docid)
        self.store.executeSQL(self.removeSQL, (docid,))