def clear_cache_delete_selected(modeladmin, request, queryset):
    """
    A delete action that will invalidate cache after being called.
    """
    result = delete_selected(modeladmin, request, queryset)

    # A result of None means that the delete happened.
    if not result and hasattr(modeladmin, 'invalidate_cache'):
        modeladmin.invalidate_cache(queryset=queryset)

    return result