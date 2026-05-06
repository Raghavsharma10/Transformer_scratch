def query_sum(queryset, field):
    """
    Let the DBMS perform a sum on a queryset
    """
    return queryset.aggregate(s=models.functions.Coalesce(models.Sum(field), 0))['s']