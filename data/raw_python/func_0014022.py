def _model_class_from_pk(definition_cls, definition_pk):
    """
    Helper used to unpickle MutableModel model class from their definition
    pk.
    """
    try:
        return definition_cls.objects.get(pk=definition_pk).model_class()
    except definition_cls.DoesNotExist:
        pass