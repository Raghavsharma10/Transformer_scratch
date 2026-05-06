def replicate_attributes(source, target, cache=None):
    '''Replicates common SQLAlchemy attributes from the `source` object to the
    `target` object.'''
    target_manager = manager_of_class(type(target))
    column_attrs = set()
    relationship_attrs = set()
    relationship_columns = set()
    for attr in manager_of_class(type(source)).attributes:
        if attr.key not in target_manager:
            # It's not common attribute
            continue
        target_attr = target_manager[attr.key]
        if isinstance(attr.property, ColumnProperty):
            assert isinstance(target_attr.property, ColumnProperty)
            column_attrs.add(attr)
        elif isinstance(attr.property, RelationshipProperty):
            assert isinstance(target_attr.property, RelationshipProperty)
            relationship_attrs.add(attr)
            if attr.property.direction is MANYTOONE:
                relationship_columns.update(attr.property.local_columns)
    for attr in column_attrs:
        if _column_property_in_registry(attr.property, _excluded):
            continue
        elif (not _column_property_in_registry(attr.property, _included) and
                 all(column in relationship_columns
                     for column in attr.property.columns)):
            continue
        setattr(target, attr.key, getattr(source, attr.key))
    for attr in relationship_attrs:
        target_attr_model = target_manager[attr.key].property.argument
        if not is_relation_replicatable(attr):
            continue
        replicate_relation(source, target, attr, target_manager[attr.key],
                           cache=cache)