def update(instance, full_clean=True, post_save=False, **kwargs):
    "Atomically update instance, setting field/value pairs from kwargs"

    # apply the updated args to the instance to mimic the change
    # note that these might slightly differ from the true database values
    # as the DB could have been updated by another thread. callers should
    # retrieve a new copy of the object if up-to-date values are required
    for k, v in kwargs.iteritems():
        if isinstance(v, ExpressionNode):
            v = resolve_expression_node(instance, v)
        setattr(instance, k, v)

    # clean instance before update
    if full_clean:
        instance.full_clean()

    # fields that use auto_now=True should be updated corrected, too!
    for field in instance._meta.fields:
        if hasattr(field, 'auto_now') and field.auto_now and field.name not in kwargs:
            kwargs[field.name] = field.pre_save(instance, False)

    rows_affected = instance.__class__._default_manager.filter(
        pk=instance.pk).update(**kwargs)

    if post_save:
        signals.post_save.send(sender=instance.__class__, instance=instance)

    return rows_affected