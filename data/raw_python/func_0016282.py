def _on_model_save(sender, **kwargs):
    """Update document in search index post_save."""
    instance = kwargs.pop("instance")
    update_fields = kwargs.pop("update_fields")
    for index in instance.search_indexes:
        try:
            _update_search_index(
                instance=instance, index=index, update_fields=update_fields
            )
        except Exception:
            logger.exception("Error handling 'on_save' signal for %s", instance)