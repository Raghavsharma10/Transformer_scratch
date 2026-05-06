def _new_percolator(spec, search_pattern):
    """Create new percolator associated with the new set."""
    if spec and search_pattern:
        query = query_string_parser(search_pattern=search_pattern).to_dict()
        for index in current_search.mappings.keys():
            # Create the percolator doc_type in the existing index for >= ES5
            # TODO: Consider doing this only once in app initialization
            percolator_doc_type = _get_percolator_doc_type(index)
            _create_percolator_mapping(index, percolator_doc_type)
            current_search_client.index(
                index=index, doc_type=percolator_doc_type,
                id='oaiset-{}'.format(spec),
                body={'query': query}
            )