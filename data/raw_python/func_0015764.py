def _boosted_value(name, action, key, value, boost):
    """Boost a value if we should in _process_queries"""
    if boost is not None:
        # Note: Most queries use 'value' for the key name except
        # Match queries which use 'query'. So we have to do some
        # switcheroo for that.
        value_key = 'query' if action in MATCH_ACTIONS else 'value'
        return {name: {'boost': boost, value_key: value}}
    return {name: value}