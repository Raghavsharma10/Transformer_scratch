def unrate_url(obj):
    """
    Generates a link to "un-rate" the given object - this
    can be used as a form target or for POSTing via Ajax.
    """
    return reverse('ratings_unrate_object', args=(
        ContentType.objects.get_for_model(obj).pk,
        obj.pk,
    ))