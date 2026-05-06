def get_embeddings_index(embedding_type='glove.42B.300d', embedding_dims=None, embedding_path=None, cache=True):
    """Retrieves embeddings index from embedding name or path. Will automatically download and cache as needed.

    Args:
        embedding_type: The embedding type to load.
        embedding_path: Path to a local embedding to use instead of the embedding type. Ignores `embedding_type` if specified.

    Returns:
        The embeddings indexed by word.
    """

    if embedding_path is not None:
        embedding_type = embedding_path  # identify embedding by path

    embeddings_index = _EMBEDDINGS_CACHE.get(embedding_type)
    if embeddings_index is not None:
        return embeddings_index

    if embedding_path is None:
        embedding_type_obj = get_embedding_type(embedding_type)

        # some very rough wrangling of zip files with the keras util `get_file`
        # a special problem: when multiple files are in one zip file
        extract = embedding_type_obj.get('extract', True)
        file_path = get_file(
            embedding_type_obj['file'], origin=embedding_type_obj['url'], extract=extract, cache_subdir='embeddings', file_hash=embedding_type_obj.get('file_hash',))

        if 'file_in_zip' in embedding_type_obj:
            zip_folder = file_path.split('.zip')[0]
            with ZipFile(file_path, 'r') as zf:
                zf.extractall(zip_folder)
            file_path = os.path.join(
                zip_folder, embedding_type_obj['file_in_zip'])
        else:
            if extract:
                if file_path.endswith('.zip'):
                    file_path = file_path.split('.zip')[0]
                # if file_path.endswith('.gz'):
                #     file_path = file_path.split('.gz')[0]
    else:
        file_path = embedding_path

    embeddings_index = _build_embeddings_index(file_path, embedding_dims)

    if cache:
        _EMBEDDINGS_CACHE[embedding_type] = embeddings_index
    return embeddings_index