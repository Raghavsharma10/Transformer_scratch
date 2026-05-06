def generic_annotate(qs_model, generic_qs_model, aggregator, gfk_field=None, alias='score'):
    """
    Find blog entries with the most comments:
    
        qs = generic_annotate(Entry.objects.public(), Comment.objects.public(), Count('comments__id'))
        for entry in qs:
            print entry.title, entry.score
    
    Find the highest rated foods:
    
        generic_annotate(Food, Rating, Avg('ratings__rating'), alias='avg')
        for food in qs:
            print food.name, '- average rating:', food.avg
    
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
    :param alias: attribute name to use for annotation
    """
    return fallback_generic_annotate(qs_model, generic_qs_model, aggregator, gfk_field, alias)