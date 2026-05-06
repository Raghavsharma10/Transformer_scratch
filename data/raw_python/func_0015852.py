def delete_entry(sender, instance, **kwargs):
    """
    Deletes Entry instance corresponding to specified instance.

    :param sender: the sending class.
    :param instance: the instance being deleted.
    """
    from ..models import Entry

    Entry.objects.get_for_model(instance)[0].delete()