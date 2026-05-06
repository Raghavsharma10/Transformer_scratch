def set_doc_ids(self, doc_ids):
        """ Build xml documents from a list of document ids.

            Args:
                doc_ids -- A document id or a lost of those.
        """
        if isinstance(doc_ids, list):
            self.set_documents(dict.fromkeys(doc_ids))
        else:
            self.set_documents({doc_ids: None})