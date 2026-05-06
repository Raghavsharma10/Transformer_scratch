def index_search_document(self, *, index):
        """
        Create or replace search document in named index.

        Checks the local cache to see if the document has changed,
        and if not aborts the update, else pushes to ES, and then
        resets the local cache. Cache timeout is set as "cache_expiry"
        in the settings, and defaults to 60s.

        """
        cache_key = self.search_document_cache_key
        new_doc = self.as_search_document(index=index)
        cached_doc = cache.get(cache_key)
        if new_doc == cached_doc:
            logger.debug("Search document for %r is unchanged, ignoring update.", self)
            return []
        cache.set(cache_key, new_doc, timeout=get_setting("cache_expiry", 60))
        get_client().index(
            index=index, doc_type=self.search_doc_type, body=new_doc, id=self.pk
        )