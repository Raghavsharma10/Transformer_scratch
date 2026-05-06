def unindex_objects(mapping_type, ids, es=None, index=None):
    """Remove documents of a specified mapping_type from the index.

    This allows for asynchronous deleting.

    If a mapping_type extends Indexable, you can add a ``pre_delete``
    hook for the model that it's based on like this::

        @receiver(dbsignals.pre_delete, sender=MyModel)
        def remove_from_index(sender, instance, **kw):
            from elasticutils.contrib.django import tasks
            tasks.unindex_objects.delay(MyMappingType, [instance.id])

    :arg mapping_type: the mapping type for these ids
    :arg ids: the list of ids of things to remove
    :arg es: The `Elasticsearch` to use. If you don't specify an
        `Elasticsearch`, it'll use `mapping_type.get_es()`.
    :arg index: The name of the index to use. If you don't specify one
        it'll use `mapping_type.get_index()`.
    """
    if settings.ES_DISABLED:
        return

    for id_ in ids:
        mapping_type.unindex(id_, es=es, index=index)