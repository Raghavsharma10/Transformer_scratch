def remove(self, obj_or_string, commit=True):
        """
        Removes an object from the index.
        :param obj_or_string:
        :param commit:
        """
        if not self.setup_complete:
            try:
                self.setup()
            except elasticsearch.TransportError as e:
                if not self.silently_fail:
                    raise
                doc_id = get_identifier(obj_or_string)
                self.log.error("Failed to remove document '%s' from Elasticsearch: %s", doc_id, e)
                return

        for language in self.languages:
            # self.log.debug('removing {0} from index {1}'.format(obj_or_string, language))
            self.index_name = self._index_name_for_language(language)
            with translation.override(language):
                super(ElasticsearchMultilingualSearchBackend, self).remove(obj_or_string,
                                                                           commit=commit)