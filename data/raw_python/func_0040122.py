def get_nested_dicts_with_key_value(parent_dict: dict, key, value):
    """Return all nested dictionaries that contain a key with a specific value. A sub-case of NestedLookup."""
    references = []
    NestedLookup(parent_dict, references, NestedLookup.key_value_equality_factory(key, value))
    return (document for document, _ in references)