def _get_named_graph(context):
    """
    Returns the named graph for this context.
    """
    if context is None:
        return None

    return models.NamedGraph.objects.get_or_create(identifier=context.identifier)[0]