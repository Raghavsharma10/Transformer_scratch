def refresh_index(cls, es=None, index=None):
        """Refreshes the index.

        Elasticsearch will update the index periodically
        automatically. If you need to see the documents you just
        indexed in your search results right now, you should call
        `refresh_index` as soon as you're done indexing. This is
        particularly helpful for unit tests.

        :arg es: The `Elasticsearch` to use. If you don't specify an
            `Elasticsearch`, it'll use `cls.get_es()`.

        :arg index: The name of the index to use. If you don't specify one
            it'll use `cls.get_index()`.

        """
        if es is None:
            es = cls.get_es()

        if index is None:
            index = cls.get_index()

        es.indices.refresh(index=index)