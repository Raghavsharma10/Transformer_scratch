def clear(self, models=None, commit=True):
        """
        Clears all indexes for the current project.
        :param models: if specified, only deletes the entries for the given models.
        :param commit: This is ignored by Haystack (maybe a bug?)
        """
        for language in self.languages:
            self.log.debug('clearing index for {0}'.format(language))
            self.index_name = self._index_name_for_language(language)
            super(ElasticsearchMultilingualSearchBackend, self).clear(models, commit)
        self._reset_existing_mapping()