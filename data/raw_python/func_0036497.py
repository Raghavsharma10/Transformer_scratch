def rate_url(obj, score=1):
    """
    Generates a link to "rate" the given object with the provided score - this
    can be used as a form target or for POSTing via Ajax.
    """
    return reverse('ratings_rate_object', args=(
        ContentType.objects.get_for_model(obj).pk,
        obj.pk,
        score,
    ))