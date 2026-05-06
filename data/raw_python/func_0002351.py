def on_delete_model_translation(instance, **kwargs):
    """
    Make sure ContentItems are deleted when a translation in deleted.
    """
    translation = instance

    parent_object = translation.master
    parent_object.set_current_language(translation.language_code)

    # Also delete any associated plugins
    # Placeholders are shared between languages, so these are not affected.
    for item in ContentItem.objects.parent(parent_object, limit_parent_language=True):
        item.delete()