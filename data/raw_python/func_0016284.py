def _in_search_queryset(*, instance, index) -> bool:
    """Wrapper around the instance manager method."""
    try:
        return instance.__class__.objects.in_search_queryset(instance.id, index=index)
    except Exception:
        logger.exception("Error checking object in_search_queryset.")
        return False