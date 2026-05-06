def include(prop):
    '''Replicate property that is normally not replicated. Right now it's
    meaningful for one-to-many relations only.'''
    if isinstance(prop, QueryableAttribute):
        prop = prop.property
    assert isinstance(prop, (Column, ColumnProperty, RelationshipProperty))
    #assert isinstance(prop, RelationshipProperty)
    _included.add(prop)