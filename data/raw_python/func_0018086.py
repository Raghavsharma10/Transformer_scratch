def instantiate_all(graph):
    """
    Instantiate all ObjectProxy objects in a nested hierarchy.

    Parameters
    ----------
    graph : dict or object
        A dictionary (or an ObjectProxy) containing the object graph
        loaded from a YAML file.

    Returns
    -------
    graph : dict or object
        The dictionary or object resulting after the recursive instantiation.
    """

    def should_instantiate(obj):
        classes = [ObjectProxy, dict, list]
        return True in [isinstance(obj, cls) for cls in classes]

    if not isinstance(graph, list):
        for key in graph:
            if should_instantiate(graph[key]):
                graph[key] = instantiate_all(graph[key])
        if hasattr(graph, 'keys'):
            for key in graph.keys():
                if should_instantiate(key):
                    new_key = instantiate_all(key)
                    graph[new_key] = graph[key]
                    del graph[key]

    if isinstance(graph, ObjectProxy):
        graph = graph.instantiate()

    if isinstance(graph, list):
        for i, elem in enumerate(graph):
            if should_instantiate(elem):
                graph[i] = instantiate_all(elem)

    return graph