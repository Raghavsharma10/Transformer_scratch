def _save_percolator(self):
        """saves the query field as an elasticsearch percolator
        """
        index = Content.search_objects.mapping.index
        query_filter = self.get_content().to_dict()

        q = {}

        if "query" in query_filter:
            q = {"query": query_filter.get("query", {})}
        else:
            return

        es.index(
            index=index,
            doc_type=".percolator",
            body=q,
            id=self.es_id
        )