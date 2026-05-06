def nonraw_instance(receiver):
    """
    A signal receiver decorator that fetch the complete instance from db when
    it's passed as raw
    """
    @wraps(receiver)
    def wrapper(sender, instance, raw, using, **kwargs):
        if raw:
            instance = sender._default_manager.using(using).get(pk=instance.pk)
        return receiver(sender=sender, raw=raw, instance=instance, using=using,
                        **kwargs)
    return wrapper