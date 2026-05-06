def on_doctree_read(app, document):
    """
    Hooks into Sphinx's ``doctree-read`` event.
    """
    cache: Dict[type, Dict[str, object]] = {}
    for desc_node in document.traverse(addnodes.desc):
        if desc_node.get("domain") != "py":
            continue
        signature_node = desc_node.traverse(addnodes.desc_signature)[0]
        module_name = signature_node.get("module")
        object_name = signature_node.get("fullname")
        object_type = desc_node.get("objtype")
        module = importlib.import_module(module_name)
        if object_type == "class":
            handle_class(signature_node, module, object_name, cache)
        elif object_type in ("method", "attribute", "staticmethod", "classmethod"):
            handle_method(signature_node, module, object_name, cache)