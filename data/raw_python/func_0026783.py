def get_field_definition(node):
    """"node is a class attribute that is a mongoengine. Returns
     the definition statement for the attribute
    """

    name = node.attrname
    cls = get_node_parent_class(node)
    definition = cls.lookup(name)[1][0].statement()
    return definition