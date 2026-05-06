def sanitize_order(model):
    """
    Sanitize order values so eliminate conflicts and gaps.
    XXX: Early start, very ugly, needs work.
    """
    to_order_dict = {}

    order_field_names = []
    for field in model._meta.fields:
        if isinstance(field, models.IntegerField):
            order_field_names.append(field.name)

    for field_name in order_field_names:
        to_order_dict[field_name] = list(model.objects.all().order_by(\
                field_name, '-timestamp'))

    updates = {}
    for field_name, object_list in to_order_dict.items():
        for i, obj in enumerate(object_list):
            position = i + 1
            if getattr(obj, field_name) != position:
                if obj in updates:
                    updates[obj][field_name] = position
                else:
                    updates[obj] = {field_name: position}

    for obj, fields in updates.items():
        for field, value in fields.items():
            setattr(obj, field, value)
        obj.save()