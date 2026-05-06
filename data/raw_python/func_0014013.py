def base_definition_pre_delete(sender, instance, **kwargs):
    """
    This is used to pass data required for deletion to the post_delete
    signal that is no more available thereafter.
    """
    # see CASCADE_MARK_ORIGIN's docstring
    cascade_deletion_origin = popattr(
        instance._state, '_cascade_deletion_origin', None
    )
    if cascade_deletion_origin == 'model_def':
        return
    if (instance.base and issubclass(instance.base, models.Model) and
            instance.base._meta.abstract):
        instance._state._deletion = instance.model_def.model_class().render_state()