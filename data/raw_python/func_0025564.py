def get_subscribers(obj):
    """
    Returns the subscribers for a given object.

    :param obj: Any object.

    """
    ctype = ContentType.objects.get_for_model(obj)
    return Subscription.objects.filter(content_type=ctype, object_id=obj.pk)