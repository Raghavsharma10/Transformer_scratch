def delete_search_document(self, *, index):
        """Delete document from named index."""
        cache.delete(self.search_document_cache_key)
        get_client().delete(index=index, doc_type=self.search_doc_type, id=self.pk)