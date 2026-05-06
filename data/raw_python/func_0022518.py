def solve_bind(expr, vars):
    """Build a RowTuple from key/value pairs under the bind.

    The Bind subtree is arranged as follows:

    Bind
    | First KV Pair
    | | First Key Expression
    | | First Value Expression
    | Second KV Pair
    | | Second Key Expression
    | | Second Value Expression
    Etc...

    As we evaluate the subtree, each subsequent KV pair is evaluated with
    the all previous bingings already in scope. For example:

    bind(x: 5, y: x + 5)  # Will bind y = 10 because x is already available.
    """
    value_expressions = []
    keys = []
    for pair in expr.children:
        keys.append(solve(pair.key, vars).value)
        value_expressions.append(pair.value)

    result = row_tuple.RowTuple(ordered_columns=keys)
    intermediate_scope = scope.ScopeStack(vars, result)

    for idx, value_expression in enumerate(value_expressions):
        value = solve(value_expression, intermediate_scope).value
        # Update the intermediate bindings so as to make earlier bindings
        # already available to the next child-expression.
        result[keys[idx]] = value

    return Result(result, ())