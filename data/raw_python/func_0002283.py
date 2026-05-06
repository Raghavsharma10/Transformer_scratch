def get_parent_lookup_kwargs(parent_object):
    """
    Return lookup arguments for the generic ``parent_type`` / ``parent_id`` fields.

    :param parent_object: The parent object.
    :type parent_object: :class:`~django.db.models.Model`
    """
    if parent_object is None:
        return dict(
            parent_type__isnull=True,
            parent_id=0
        )
    elif isinstance(parent_object, models.Model):
        return dict(
            parent_type=ContentType.objects.get_for_model(parent_object),
            parent_id=parent_object.pk
        )
    else:
        raise ValueError("parent_object is not a model!")