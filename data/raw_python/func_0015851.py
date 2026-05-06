def delete_entry_tag(sender, instance, **kwargs):
    """
    Deletes EntryTag for Entry corresponding to specified
    TaggedItemBase instance.

    :param sender: the sending TaggedItemBase class.
    :param instance: the TaggedItemBase instance.
    """
    from ..models import (
        Entry,
        EntryTag
    )

    entry   = Entry.objects.get_for_model(instance.content_object)[0]
    tag     = instance.tag

    EntryTag.objects.filter(tag=tag, entry=entry).delete()