def is_valid_superset(actual_projection, index):
    """Returns True if the actual index is a valid superset of the expected index"""
    projection_type = actual_projection["ProjectionType"]
    if projection_type == "ALL":
        return True
    meta = index.model.Meta
    # all index types provide index keys and model keys
    provides = set.union(meta.keys, index.keys)
    if projection_type == "KEYS_ONLY":
        pass
    elif projection_type == "INCLUDE":  # pragma: no branch (unknown projections break loud)
        by_dynamo_name = {column.dynamo_name: column for column in meta.columns}
        provides.update(
            by_dynamo_name[name]
            for name in actual_projection["NonKeyAttributes"]
            if name in by_dynamo_name  # ignore columns the projection provides if the model doesn't care about them
        )
    else:
        logger.info(f"unexpected index ProjectionType '{projection_type}'")
        return False
    expects = index.projection["included"]
    return provides.issuperset(expects)