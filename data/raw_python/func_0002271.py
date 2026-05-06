def get_placeholder_cache_key_for_parent(parent_object, placeholder_name, language_code):
    """
    Return a cache key for a placeholder.

    This key is used to cache the entire output of a placeholder.
    """
    parent_type = ContentType.objects.get_for_model(parent_object)
    return _get_placeholder_cache_key_for_id(
        parent_type.id,
        parent_object.pk,
        placeholder_name,
        language_code
    )