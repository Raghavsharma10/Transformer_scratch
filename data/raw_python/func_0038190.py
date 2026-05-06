def create_order_objects(model, order_fields):
    """
    Create order items for objects already present in the database.
    """
    for rel in model._meta.get_all_related_objects():
        rel_model = rel.model
        if rel_model.__module__ == 'order.models':

            objs = model.objects.all()
            values = {}
            for order_field in order_fields:
                order_objs = rel_model.objects.all().order_by('-%s' \
                        % order_field)
                try:
                    values[order_field] = getattr(order_objs[0], \
                            order_field) + 1
                except IndexError:
                    values[order_field] = 1
            for obj in objs:
                try:
                    rel_model.objects.get(item=obj)
                except rel_model.DoesNotExist:
                    rel_model.objects.create(item=obj, **values)
                    for key in values:
                        values[key] += 1