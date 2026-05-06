def _init_index(root_dir, schema, index_name):
    """ Creates new index or opens existing.

    Args:
        root_dir (str): root dir where to find or create index.
        schema (whoosh.fields.Schema): schema of the index to create or open.
        index_name (str): name of the index.

    Returns:
        tuple ((whoosh.index.FileIndex, str)): first element is index, second is index directory.
    """

    index_dir = os.path.join(root_dir, index_name)
    try:
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
            return create_in(index_dir, schema), index_dir
        else:
            return open_dir(index_dir), index_dir
    except Exception as e:
        logger.error("Init error: failed to open search index at: '{}': {} ".format(index_dir, e))
        raise