def multi_constructor(loader, tag_suffix, node):
    """
    Constructor function passed to PyYAML telling it how to construct
    objects from argument descriptions. See PyYAML documentation for
    details on the call signature.
    """
    yaml_src = yaml.serialize(node)
    mapping = loader.construct_mapping(node)
    if '.' not in tag_suffix:
        classname = tag_suffix
        rval = ObjectProxy(classname, mapping, yaml_src)
    else:
        classname = try_to_import(tag_suffix)
        rval = ObjectProxy(classname, mapping, yaml_src)

    return rval