def get_document_models():
    """Return dict of index.doc_type: model."""
    mappings = {}
    for i in get_index_names():
        for m in get_index_models(i):
            key = "%s.%s" % (i, m._meta.model_name)
            mappings[key] = m
    return mappings