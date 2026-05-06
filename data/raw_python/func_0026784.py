def get_field_embedded_doc(node):
    """Returns de ClassDef for the related embedded document in a
    embedded document field."""

    definition = get_field_definition(node)
    cls_name = definition.last_child().last_child()
    cls = next(cls_name.infer())
    return cls