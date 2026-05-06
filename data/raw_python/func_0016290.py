def get_model_index_properties(instance, index):
    """Return the list of properties specified for a model in an index."""
    mapping = get_index_mapping(index)
    doc_type = instance._meta.model_name.lower()
    return list(mapping["mappings"][doc_type]["properties"].keys())