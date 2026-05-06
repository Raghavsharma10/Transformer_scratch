def generic_aggregate(qs_model, generic_qs_model, aggregator, gfk_field=None):
    """
    Find total number of comments on blog entries:
    
        generic_aggregate(Entry.objects.public(), Comment.objects.public(), Count('comments__id'))
    
    Find the average rating for foods starting with 'a':
    
        a_foods = Food.objects.filter(name__startswith='a')
        generic_aggregate(a_foods, Rating, Avg('ratings__rating'))
    
    .. note::
        In both of the above examples it is assumed that a GenericRelation exists
        on Entry to Comment (named "comments") and also on Food to Rating (named "ratings").
        If a GenericRelation does *not* exist, the query will still return correct
        results but the code path will be different as it will use the fallback method.
    
    .. warning::
        If the underlying column type differs between the qs_model's primary
        key and the generic_qs_model's foreign key column, it will use the fallback
        method, which can correctly CASTself.

    :param qs_model: A model or a queryset of objects you want to perform
        annotation on, e.g. blog entries
    :param generic_qs_model: A model or queryset containing a GFK, e.g. comments
    :param aggregator: an aggregation, from django.db.models, e.g. Count('id') or Avg('rating')
    :param gfk_field: explicitly specify the field w/the gfk
    """
    return fallback_generic_aggregate(qs_model, generic_qs_model, aggregator, gfk_field)