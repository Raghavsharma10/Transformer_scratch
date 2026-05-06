def clear_commentarea_cache(comment):
    """
    Clean the plugin output cache of a rendered plugin.
    """
    parent = comment.content_object
    for instance in CommentsAreaItem.objects.parent(parent):
        instance.clear_cache()