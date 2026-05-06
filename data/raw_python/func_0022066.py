def register(model, field_name=None, related_name=None, lookup_method_name='get_follows'):
    """
    This registers any model class to be follow-able.
    
    """
    if model in registry:
        return

    registry.append(model)
    
    if not field_name:
        field_name = 'target_%s' % model._meta.module_name
    
    if not related_name:
        related_name = 'follow_%s' % model._meta.module_name
    
    field = ForeignKey(model, related_name=related_name, null=True,
        blank=True, db_index=True)
    
    field.contribute_to_class(Follow, field_name)
    setattr(model, lookup_method_name, get_followers_for_object)
    model_map[model] = [related_name, field_name]