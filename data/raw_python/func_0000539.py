def handle_method(signature_node, module, object_name, cache):
    """
    Styles ``automethod`` entries.

    Adds ``abstract`` prefix to abstract methods.

    Adds link to originating class for inherited methods.
    """
    *class_names, attr_name = object_name.split(".")  # Handle nested classes
    class_ = module
    for class_name in class_names:
        class_ = getattr(class_, class_name, None)
        if class_ is None:
            return
    attr = getattr(class_, attr_name)
    try:
        inspected_attr = cache[class_][attr_name]
        defining_class = inspected_attr.defining_class
    except KeyError:
        # TODO: This is a hack to handle bad interaction between enum and inspect
        defining_class = class_
    if defining_class is not class_:
        reftarget = "{}.{}".format(defining_class.__module__, defining_class.__name__)
        xref_node = addnodes.pending_xref(
            "", refdomain="py", refexplicit=True, reftype="class", reftarget=reftarget
        )
        name_node = nodes.literal(
            "", "{}".format(defining_class.__name__), classes=["descclassname"]
        )
        xref_node.append(name_node)
        desc_annotation = list(signature_node.traverse(addnodes.desc_annotation))
        index = len(desc_annotation)
        class_annotation = addnodes.desc_addname()
        class_annotation.extend([nodes.Text("("), xref_node, nodes.Text(").")])
        class_annotation["xml:space"] = "preserve"
        signature_node.insert(index, class_annotation)
    else:
        is_overridden = False
        for class_ in defining_class.__mro__[1:]:
            if hasattr(class_, attr_name):
                is_overridden = True
        if is_overridden:
            emphasis = nodes.emphasis(
                "overridden ", "overridden ", classes=["property"]
            )
            signature_node.insert(0, emphasis)
    if getattr(attr, "__isabstractmethod__", False):
        emphasis = nodes.emphasis("abstract", "abstract", classes=["property"])
        signature_node.insert(0, emphasis)