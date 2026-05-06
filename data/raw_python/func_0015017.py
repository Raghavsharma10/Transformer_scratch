def _get_relationships(model):
    """
    Gets the necessary relationships for the resource
    by inspecting the sqlalchemy model for relationships.

    :param DeclarativeMeta model: The SQLAlchemy ORM model.
    :return: A tuple of Relationship/ListRelationship instances
        corresponding to the relationships on the Model.
    :rtype: tuple
    """
    relationships = []
    for name, relationship in inspect(model).relationships.items():
        class_ = relationship.mapper.class_
        if relationship.uselist:
            rel = ListRelationship(name, relation=class_.__name__)
        else:
            rel = Relationship(name, relation=class_.__name__)
        relationships.append(rel)
    return tuple(relationships)