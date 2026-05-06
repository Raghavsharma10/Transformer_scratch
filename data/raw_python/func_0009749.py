def get_route_from_solution(solution, graph):
    """Converts a solution (a list of node indices) into a list
    of paths suitable for rendering."""

    # As a guard against comparing invalid "solutions",
    # ensure that this solution is valid.
    assert check_valid_solution(solution, graph)

    return [graph.get_path(i) for i in solution]