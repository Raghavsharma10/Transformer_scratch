def order_by_position(qs, reverse=False):
    """Template filter to return a position-ordered queryset."""
    if qs:
        # ATTENTION: Django creates an invalid sql statement if two related
        # models have both generic positions, so we cannot use
        # qs.oder_by('generic_position__position')
        position = 'position'
        if reverse:
            position = '-' + position
        # Get content type of first queryset item
        c_type = ContentType.objects.get_for_model(qs[0])
        # Check that every item has a valid position item
        for obj in qs:
            ObjectPosition.objects.get_or_create(
                content_type=c_type, object_id=obj.pk)
        return [
            o.content_object for o in ObjectPosition.objects.filter(
                content_type=c_type, object_id__in=qs).order_by(position)
        ]
    return qs