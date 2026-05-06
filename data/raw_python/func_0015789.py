def bulk_index(cls, documents, id_field='id', es=None, index=None):
        """Adds or updates a batch of documents.

        :arg documents: List of Python dicts representing individual
            documents to be added to the index

            .. Note::

               This must be serializable into JSON.

        :arg id_field: The name of the field to use as the document
            id. This defaults to 'id'.

        :arg es: The `Elasticsearch` to use. If you don't specify an
            `Elasticsearch`, it'll use `cls.get_es()`.

        :arg index: The name of the index to use. If you don't specify one
            it'll use `cls.get_index()`.

        .. Note::

           If you need the documents available for searches
           immediately, make sure to refresh the index by calling
           ``refresh_index()``.

        """
        if es is None:
            es = cls.get_es()

        if index is None:
            index = cls.get_index()

        documents = (dict(d, _id=d[id_field]) for d in documents)

        bulk_index(
            es,
            documents,
            index=index,
            doc_type=cls.get_mapping_type_name(),
            raise_on_error=True
        )