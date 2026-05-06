def handle_class(signature_node, module, object_name, cache):
    """
    Styles ``autoclass`` entries.

    Adds ``abstract`` prefix to abstract classes.
    """
    class_ = getattr(module, object_name, None)
    if class_ is None:
        return
    if class_ not in cache:
        cache[class_] = {}
        attributes = inspect.classify_class_attrs(class_)
        for attribute in attributes:
            cache[class_][attribute.name] = attribute
    if inspect.isabstract(class_):
        emphasis = nodes.emphasis("abstract ", "abstract ", classes=["property"])
        signature_node.insert(0, emphasis)