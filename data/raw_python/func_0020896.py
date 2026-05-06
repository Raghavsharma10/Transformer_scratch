def _percolate_query(index, doc_type, percolator_doc_type, document):
    """Get results for a percolate query."""
    if ES_VERSION[0] in (2, 5):
        results = current_search_client.percolate(
            index=index, doc_type=doc_type, allow_no_indices=True,
            ignore_unavailable=True, body={'doc': document}
        )
        return results['matches']
    elif ES_VERSION[0] == 6:
        results = current_search_client.search(
            index=index, doc_type=percolator_doc_type, allow_no_indices=True,
            ignore_unavailable=True, body={
                'query': {
                    'percolate': {
                        'field': 'query',
                        'document_type': percolator_doc_type,
                        'document': document,
                    }
                }
            }
        )
        return results['hits']['hits']