def base_definition_post_delete(sender, instance, **kwargs):
    """
    Make sure to delete fields inherited from an abstract model base.
    """
    if hasattr(instance._state, '_deletion'):
        # Make sure to flatten abstract bases since Django
        # migrations can't deal with them.
        model = popattr(instance._state, '_deletion')
        for field in instance.base._meta.fields:
            perform_ddl('remove_field', model, field)