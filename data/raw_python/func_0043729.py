def position_input(obj, visible=False):
    """Template tag to return an input field for the position of the object."""
    if not obj.generic_position.all():
        ObjectPosition.objects.create(content_object=obj)
    return {'obj': obj, 'visible': visible,
            'object_position': obj.generic_position.all()[0]}