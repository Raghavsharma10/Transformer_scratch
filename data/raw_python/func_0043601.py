def get_conjunctive_graph(store_id=None):
    """
    Returns an open conjunctive graph.
    """
    if not store_id:
        store_id = DEFAULT_STORE

    store = DjangoStore(DEFAULT_STORE)
    graph = ConjunctiveGraph(store=store, identifier=store_id)
    if graph.open(None) != VALID_STORE:
        raise ValueError("The store identified by {0} is not a valid store".format(store_id))
    return graph