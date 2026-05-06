def iter_columns(condition):
    """
    Yield all columns in the condition or its inner conditions.

    Unwraps proxies when the condition's column (or any of its values) include paths.
    """
    # Like iter_conditions, this can't live in each condition without going possibly infinite on the
    # recursion, or passing the visited set through every call.  That makes the signature ugly, so we
    # take care of it here.  Luckily, it's pretty easy to leverage iter_conditions and just unpack the
    # actual columns.
    visited = set()
    for condition in iter_conditions(condition):
        if condition.operation in ("and", "or", "not"):
            continue
        # Non-meta conditions always have a column, and each of values has the potential to be a column.
        # Comparison will only have a list of len 1, but it's simpler to just iterate values and check each

        # unwrap proxies created for paths
        column = proxied(condition.column)

        # special case for None
        # this could also have skipped on isinstance(condition, Condition)
        # but this is slightly more flexible for users to create their own None-sentinel Conditions
        if column is None:
            continue
        if column not in visited:
            visited.add(column)
            yield column
            for value in condition.values:
                if isinstance(value, ComparisonMixin):
                    if value not in visited:
                        visited.add(value)
                        yield value