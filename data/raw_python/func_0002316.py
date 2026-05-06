def get_cached_placeholder_output(parent_object, placeholder_name):
    """
    Return cached output for a placeholder, if available.
    This avoids fetching the Placeholder object.
    """
    if not PlaceholderRenderingPipe.may_cache_placeholders():
        return None

    language_code = get_parent_language_code(parent_object)
    cache_key = get_placeholder_cache_key_for_parent(parent_object, placeholder_name, language_code)
    return cache.get(cache_key)