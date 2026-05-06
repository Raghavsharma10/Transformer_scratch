def exclude(prop):
    '''Don't replicate property that is normally replicated: ordering column,
    many-to-one relation that is marked for replication from other side.'''
    if isinstance(prop, QueryableAttribute):
        prop = prop.property
    assert isinstance(prop, (Column, ColumnProperty, RelationshipProperty))
    _excluded.add(prop)
    if isinstance(prop, RelationshipProperty):
        # Also exclude columns that participate in this relationship
        for local in prop.local_columns:
            _excluded.add(local)