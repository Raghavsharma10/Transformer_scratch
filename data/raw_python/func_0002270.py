def get_placeholder_cache_key(placeholder, language_code):
    """
    Return a cache key for an existing placeholder object.

    This key is used to cache the entire output of a placeholder.
    """
    return _get_placeholder_cache_key_for_id(
        placeholder.parent_type_id,
        placeholder.parent_id,
        placeholder.slot,
        language_code
    )