def update(self, doc: dict, doc_id: str):
        """Partial update to a single document.

        Uses the Update API with the specified partial document.
        """
        body = {
            'doc': doc
        }
        self.instance.update(self.index, self.doc_type, doc_id, body=body)