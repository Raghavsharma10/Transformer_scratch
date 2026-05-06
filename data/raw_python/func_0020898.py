def _delete_percolator(spec, search_pattern):
    """Delete percolator associated with the new oaiset."""
    if spec:
        for index in current_search.mappings.keys():
            # Create the percolator doc_type in the existing index for >= ES5
            percolator_doc_type = _get_percolator_doc_type(index)
            _create_percolator_mapping(index, percolator_doc_type)
            current_search_client.delete(
                index=index, doc_type=percolator_doc_type,
                id='oaiset-{}'.format(spec), ignore=[404]
            )