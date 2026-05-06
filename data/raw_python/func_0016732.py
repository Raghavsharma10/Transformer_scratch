def iter_conditions(condition):
    """Yield all conditions within the given condition.

    If the root condition is and/or/not, it is not yielded (unless a cyclic reference to it is found)."""
    conditions = list()
    visited = set()
    # Has to be split out, since we don't want to visit the root (for cyclic conditions)
    # but we don't want to yield it (if it's non-cyclic) because this only yields inner conditions
    if condition.operation in {"and", "or"}:
        conditions.extend(reversed(condition.values))
    elif condition.operation == "not":
        conditions.append(condition.values[0])
    else:
        conditions.append(condition)
    while conditions:
        condition = conditions.pop()
        if condition in visited:
            continue
        visited.add(condition)
        yield condition
        if condition.operation in {"and", "or", "not"}:
            conditions.extend(reversed(condition.values))