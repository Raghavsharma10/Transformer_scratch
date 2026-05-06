def faceted_search(cls, query=None, filters=None, search=None):
        """Return faceted search instance with defaults set.

        :param query: Elastic DSL query object (``Q``).
        :param filters: Dictionary with selected facet values.
        :param search: An instance of ``Search`` class. (default: ``cls()``).
        """
        search_ = search or cls()

        class RecordsFacetedSearch(FacetedSearch):
            """Pass defaults from ``cls.Meta`` object."""

            index = prefix_index(app=current_app, index=search_._index[0])
            doc_types = getattr(search_.Meta, 'doc_types', ['_all'])
            fields = getattr(search_.Meta, 'fields', ('*', ))
            facets = getattr(search_.Meta, 'facets', {})

            def search(self):
                """Use ``search`` or ``cls()`` instead of default Search."""
                # Later versions of `elasticsearch-dsl` (>=5.1.0) changed the
                # Elasticsearch FacetedResponse class constructor signature.

                if ES_VERSION[0] > 2:
                    return search_.response_class(FacetedResponse)
                return search_.response_class(partial(FacetedResponse, self))

        return RecordsFacetedSearch(query=query, filters=filters or {})