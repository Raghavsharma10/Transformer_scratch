def serialize_instance(instance):
    """ Serialize Django model instance """
    model_name = force_text(instance._meta)
    return '{}:{}'.format(model_name, instance.pk)