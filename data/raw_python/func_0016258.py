def as_search_action(self, *, index, action):
        """
        Return an object as represented in a bulk api operation.

        Bulk API operations have a very specific format. This function will
        call the standard `as_search_document` method on the object and then
        wrap that up in the correct format for the action specified.

        https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html

        Args:
            index: string, the name of the index in which the action is to
                be taken. Bulk operations are only every carried out on a single
                index at a time.
            action: string ['index' | 'update' | 'delete'] - this decides
                how the final document is formatted.

        Returns a dictionary.

        """
        if action not in ("index", "update", "delete"):
            raise ValueError("Action must be 'index', 'update' or 'delete'.")

        document = {
            "_index": index,
            "_type": self.search_doc_type,
            "_op_type": action,
            "_id": self.pk,
        }

        if action == "index":
            document["_source"] = self.as_search_document(index=index)
        elif action == "update":
            document["doc"] = self.as_search_document(index=index)
        return document