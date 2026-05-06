def bfs(graph, func, head, reverse=None):
    """
    BREADTH FIRST SEARCH

    IF func RETURNS FALSE, THEN NO MORE PATHS DOWN THE BRANCH ARE TAKEN

    IT'S EXPECTED func TAKES THESE ARGUMENTS:
    node - THE CURRENT NODE IN THE
    path - PATH FROM head TO node
    graph - THE WHOLE GRAPH
    todo - WHAT'S IN THE QUEUE TO BE DONE
    """

    todo = deque()  # LIST OF PATHS
    todo.append(Step(None, head))

    while todo:
        path = todo.popleft()
        keep_going = func(path.node, Path(path), graph, todo)
        if keep_going:
            todo.extend(Step(path, c) for c in graph.get_children(path.node))