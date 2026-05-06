def content_deleted(sender, instance=None, **kwargs):
    """removes content from the ES index when deleted from DB
    """
    if getattr(instance, "_index", True):
        cls = instance.get_real_instance_class()
        index = cls.search_objects.mapping.index
        doc_type = cls.search_objects.mapping.doc_type

        cls.search_objects.client.delete(index, doc_type, instance.id, ignore=[404])