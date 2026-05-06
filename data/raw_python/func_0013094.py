def _get_by_id_async(cls, id, parent=None, app=None, namespace=None,
                       **ctx_options):
    """Returns an instance of Model class by ID (and app, namespace).

    This is the asynchronous version of Model._get_by_id().
    """
    key = Key(cls._get_kind(), id, parent=parent, app=app, namespace=namespace)
    return key.get_async(**ctx_options)