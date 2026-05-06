def delete_collection_percolator(target):
    """Delete percolator associated with the new collection.

    :param target: Collection where the percolator was attached.
    """
    for name in current_search.mappings.keys():
        if target.name and target.dbquery:
            current_search.client.delete(
                index=name,
                doc_type='.percolator',
                id='collection-{}'.format(target.name),
                ignore=[404]
            )