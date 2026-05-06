def _on_model_delete(sender, **kwargs):
    """Remove documents from search indexes post_delete."""
    instance = kwargs.pop("instance")
    for index in instance.search_indexes:
        try:
            _delete_from_search_index(instance=instance, index=index)
        except Exception:
            logger.exception("Error handling 'on_delete' signal for %s", instance)