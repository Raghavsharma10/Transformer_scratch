def reindex(self, new_index_name: str, identifier_key: str, **kwargs) -> 'ElasticIndex':
        """Reindex the entire index.

        Scrolls the old index and bulk indexes all data into the new index.

        :param new_index_name:
        :param identifier_key:
        :param kwargs:          Overwrite ElasticIndex __init__ params.
        :return:
        """
        if 'url' not in kwargs:
            kwargs['url'] = self.url
        if 'doc_type' not in kwargs:
            kwargs['doc_type'] = self.doc_type
        if 'mapping' not in kwargs:
            kwargs['mapping'] = self.mapping
        new_index = ElasticIndex(new_index_name, **kwargs)

        for results in self.scroll(size=500):
            new_index.bulk(results, identifier_key)
        return new_index