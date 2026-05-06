def _update_search_index(*, instance, index, update_fields):
    """Process index / update search index update actions."""
    if not _in_search_queryset(instance=instance, index=index):
        logger.debug(
            "Object (%r) is not in search queryset, ignoring update.", instance
        )
        return

    try:
        if update_fields:
            pre_update.send(
                sender=instance.__class__,
                instance=instance,
                index=index,
                update_fields=update_fields,
            )
            if settings.auto_sync(instance):
                instance.update_search_document(
                    index=index, update_fields=update_fields
                )
        else:
            pre_index.send(sender=instance.__class__, instance=instance, index=index)
            if settings.auto_sync(instance):
                instance.index_search_document(index=index)
    except Exception:
        logger.exception("Error handling 'post_save' signal for %s", instance)