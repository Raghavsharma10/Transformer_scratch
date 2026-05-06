def new_collection_percolator(target):
    """Create new percolator associated with the new collection.

    :param target: Collection where the percolator will be atached.
    """
    query = IQ(target.dbquery)
    for name in current_search.mappings.keys():
        if target.name and target.dbquery:
            current_search.client.index(
                index=name,
                doc_type='.percolator',
                id='collection-{}'.format(target.name),
                body={'query': query.to_dict()}
            )