def update_entry_attributes(sender, instance, **kwargs):
    """
    Updates attributes for Entry instance corresponding to
    specified instance.

    :param sender: the sending class.
    :param instance: the instance being saved.
    """
    from ..models import Entry

    entry = Entry.objects.get_for_model(instance)[0]

    default_url = getattr(instance, 'get_absolute_url', '')
    entry.title = getattr(instance, 'title', str(instance))
    entry.url   = getattr(instance, 'url', default_url)
    entry.live  = bool(getattr(instance, 'live', True))

    entry.save()