def update_search_document(self, *, index, update_fields):
        """
        Partial update of a document in named index.

        Partial updates are invoked via a call to save the document
        with 'update_fields'. These fields are passed to the
        as_search_document method so that it can build a partial
        document. NB we don't just call as_search_document and then
        strip the fields _not_ in update_fields as we are trying
        to avoid possibly expensive operations in building the
        source document. The canonical example for this method
        is updating a single timestamp on a model - we don't want
        to have to walk the model relations and build a document
        in this case - we just want to push the timestamp.

        When POSTing a partial update the `as_search_document` doc
        must be passed to the `client.update` wrapped in a "doc" node,
        see: https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-update.html

        """
        doc = self.as_search_document_update(index=index, update_fields=update_fields)
        if not doc:
            logger.debug("Ignoring object update as document is empty.")
            return

        get_client().update(
            index=index, doc_type=self.search_doc_type, body={"doc": doc}, id=self.pk
        )