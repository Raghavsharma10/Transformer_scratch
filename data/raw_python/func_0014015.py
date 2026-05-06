def raw_field_definition_proxy_post_save(sender, instance, raw, **kwargs):
    """
    When proxy field definitions are loaded from a fixture they're not
    passing through the `field_definition_post_save` signal. Make sure they
    are.
    """
    if raw:
        model_class = instance.content_type.model_class()
        opts = model_class._meta
        if opts.proxy and opts.concrete_model is sender:
            field_definition_post_save(
                sender=model_class, instance=instance.type_cast(), raw=raw,
                **kwargs
            )