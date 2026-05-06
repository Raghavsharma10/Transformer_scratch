def save_positions(post_data, queryset=None):
    """
    Function to update a queryset of position objects with a post data dict.

    :post_data: Typical post data dictionary like ``request.POST``, which
      contains the keys of the position inputs.
    :queryset: Queryset of the model ``ObjectPosition``.

    """
    if not queryset:
        queryset = ObjectPosition.objects.all()
    for key in post_data:
        if key.startswith('position-'):
            try:
                obj_id = int(key.replace('position-', ''))
            except ValueError:
                continue
            queryset.filter(pk=obj_id).update(position=post_data[key])