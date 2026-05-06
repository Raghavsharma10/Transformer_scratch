def unindex(cls, id_, es=None, index=None):
        """Removes a particular item from the search index.

        :arg id_: The Elasticsearch id for the document to remove from
            the index.

        :arg es: The `Elasticsearch` to use. If you don't specify an
            `Elasticsearch`, it'll use `cls.get_es()`.

        :arg index: The name of the index to use. If you don't specify one
            it'll use `cls.get_index()`.

        """
        if es is None:
            es = cls.get_es()

        if index is None:
            index = cls.get_index()

        es.delete(index=index, doc_type=cls.get_mapping_type_name(), id=id_)