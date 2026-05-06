def generic_filter(generic_qs_model, filter_qs_model, gfk_field=None):
    """
    Only show me ratings made on foods that start with "a":
    
        a_foods = Food.objects.filter(name__startswith='a')
        generic_filter(Rating.objects.all(), a_foods)
    
    Only show me comments from entries that are marked as public:
    
        generic_filter(Comment.objects.public(), Entry.objects.public())
    
    :param generic_qs_model: A model or queryset containing a GFK, e.g. comments
    :param qs_model: A model or a queryset of objects you want to restrict the generic_qs to
    :param gfk_field: explicitly specify the field w/the gfk
    """
    generic_qs = normalize_qs_model(generic_qs_model)
    filter_qs = normalize_qs_model(filter_qs_model)
    
    if not gfk_field:
        gfk_field = get_gfk_field(generic_qs.model)
    
    pk_field_type = get_field_type(filter_qs.model._meta.pk)
    gfk_field_type = get_field_type(generic_qs.model._meta.get_field(gfk_field.fk_field))
    if pk_field_type != gfk_field_type:
        return fallback_generic_filter(generic_qs, filter_qs, gfk_field)
    
    return generic_qs.filter(**{
        gfk_field.ct_field: ContentType.objects.get_for_model(filter_qs.model),
        '%s__in' % gfk_field.fk_field: filter_qs.values('pk'),
    })