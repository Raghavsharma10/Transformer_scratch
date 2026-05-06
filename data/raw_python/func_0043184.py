def _get_query_sets_for_object(o):
    """
    Determines the correct query set based on the object.

    If the object is a literal, it will return a query set over LiteralStatements.
    If the object is a URIRef or BNode, it will return a query set over Statements.
    If the object is unknown, it will return both the LiteralStatement and Statement query sets.

    This method always returns a list of size at least one.
    """
    if o:
        if isinstance(o, Literal):
            query_sets = [models.LiteralStatement.objects]
        else:
            query_sets = [models.URIStatement.objects]
    else:
        query_sets = [models.URIStatement.objects, models.LiteralStatement.objects]
    return query_sets