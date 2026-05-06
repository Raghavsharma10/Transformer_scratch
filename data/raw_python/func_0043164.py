def update(self, index, iterable, commit=True):
        """
        Updates the index with current data.
        :param index: The search_indexes.Index object
        :param iterable: The queryset
        :param commit: commit to the backend.
        """
        parler = False
        # setup here because self.existing_mappings are overridden.
        if not self.setup_complete:
            try:
                self.setup()
            except elasticsearch.TransportError as e:
                if not self.silently_fail:
                    raise

                self.log.error("Failed to add documents to Elasticsearch: %s", e)
                return

        if hasattr(iterable, 'language') and hasattr(iterable.language, '__call__'):
            parler = True  # Django-parler

        for language in self.languages:
            self.index_name = self._index_name_for_language(language)
            # self.log.debug('updating index for {0}'.format(language))
            if parler:
                # workaround for django-parler
                for item in iterable:
                    item.set_current_language(language)
                super(ElasticsearchMultilingualSearchBackend, self).update(
                    index, iterable, commit)
            else:
                with translation.override(language):
                    super(ElasticsearchMultilingualSearchBackend, self).update(
                        index, iterable, commit)