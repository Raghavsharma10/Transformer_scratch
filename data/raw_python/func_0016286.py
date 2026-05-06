def _delete_from_search_index(*, instance, index):
    """Remove a document from a search index."""
    pre_delete.send(sender=instance.__class__, instance=instance, index=index)
    if settings.auto_sync(instance):
        instance.delete_search_document(index=index)