def multi_constructor_pkl(loader, tag_suffix, node):
    """
    Constructor function passed to PyYAML telling it how to load
    objects from paths to .pkl files. See PyYAML documentation for
    details on the call signature.
    """

    mapping = loader.construct_yaml_str(node)
    if tag_suffix != "" and tag_suffix != u"":
        raise AssertionError('Expected tag_suffix to be "" but it is "'+tag_suffix+'"')

    rval = ObjectProxy(None, {}, yaml.serialize(node))
    rval.instance = serial.load(mapping)

    return rval