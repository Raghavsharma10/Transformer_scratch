def index(cls, document, id_=None, overwrite_existing=True, es=None,
              index=None):
        """Adds or updates a document to the index

        :arg document: Python dict of key/value pairs representing
            the document

            .. Note::

               This must be serializable into JSON.

        :arg id_: the id of the document

            .. Note::

               If you don't provide an ``id_``, then Elasticsearch
               will make up an id for your document and it'll look
               like a character name from a Lovecraft novel.

        :arg overwrite_existing: if ``True`` overwrites existing documents
             of the same ID and doctype

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

        kw = {}
        if not overwrite_existing:
            kw['op_type'] = 'create'
        es.index(index=index, doc_type=cls.get_mapping_type_name(),
                 body=document, id=id_, **kw)