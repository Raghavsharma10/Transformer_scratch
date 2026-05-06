def get_named_graph(identifier, store_id=DEFAULT_STORE, create=True):
    """
    Returns an open named graph.
    """
    if not isinstance(identifier, URIRef):
        identifier = URIRef(identifier)

    store = DjangoStore(store_id)
    graph = Graph(store, identifier=identifier)
    if graph.open(None, create=create) != VALID_STORE:
        raise ValueError("The store identified by {0} is not a valid store".format(store_id))
    return graph