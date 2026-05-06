def check_valid_solution(solution, graph):
    """Check that the solution is valid: every path is visited exactly once."""
    expected = Counter(
        i for (i, _) in graph.iter_starts_with_index()
        if i < graph.get_disjoint(i)
    )
    actual = Counter(
        min(i, graph.get_disjoint(i))
        for i in solution
    )

    difference = Counter(expected)
    difference.subtract(actual)
    difference = {k: v for k, v in difference.items() if v != 0}
    if difference:
        print('Solution is not valid!'
              'Difference in node counts (expected - actual): {}'.format(difference))
        return False
    return True