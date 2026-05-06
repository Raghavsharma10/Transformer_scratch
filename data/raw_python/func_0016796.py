def check_range_key(query_on, key):
    """BeginsWith, Between, or any Comparison except '!=' against query_on.range_key"""
    return (
        isinstance(key, BaseCondition) and
        key.operation in ("begins_with", "between", "<", ">", "<=", ">=", "==") and
        key.column is query_on.range_key
    )