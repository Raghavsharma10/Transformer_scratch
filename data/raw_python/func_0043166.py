def search(self, query_string, **kwargs):
        """
        The main search method
        :param query_string: The string to pass to Elasticsearch. e.g. '*:*'
        :param kwargs: start_offset, end_offset, result_class
        :return: result_class instance
        """
        self.index_name = self._index_name_for_language(translation.get_language())
        # self.log.debug('search method called (%s): %s' %
        #                (translation.get_language(), query_string))
        return super(ElasticsearchMultilingualSearchBackend, self).search(query_string, **kwargs)