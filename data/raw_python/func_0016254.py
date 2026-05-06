def search_document_cache_key(self):
        """Key used for storing search docs in local cache."""
        return "elasticsearch_django:{}.{}.{}".format(
            self._meta.app_label, self._meta.model_name, self.pk
        )