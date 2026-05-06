def fetch_search_document(self, *, index):
        """Fetch the object's document from a search index by id."""
        assert self.pk, "Object must have a primary key before being indexed."
        client = get_client()
        return client.get(index=index, doc_type=self.search_doc_type, id=self.pk)