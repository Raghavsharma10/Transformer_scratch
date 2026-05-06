def is_subscribed(user, obj):
    """
    Returns ``True`` if the user is subscribed to the given object.

    :param user: A ``User`` instance.
    :param obj: Any object.

    """
    if not user.is_authenticated():
        return False

    ctype = ContentType.objects.get_for_model(obj)

    try:
        Subscription.objects.get(
            user=user, content_type=ctype, object_id=obj.pk)
    except Subscription.DoesNotExist:
        return False

    return True