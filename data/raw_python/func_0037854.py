def adapt_persistent_instance(persistent_object, target_rest_class=None, attribute_filter=None):
    """
    Adapts a single persistent instance to a REST model; at present this is a
    common method for all persistent backends.

    Refer to: https://groups.google.com/forum/#!topic/prestans-discuss/dO1yx8f60as
    for discussion on this feature
    """

    # try and get the adapter and the REST class for the persistent object
    if target_rest_class is None:
        adapter_instance = registry.get_adapter_for_persistent_model(persistent_object)
    else:
        if inspect.isclass(target_rest_class):
            target_rest_class = target_rest_class()

        adapter_instance = registry.get_adapter_for_persistent_model(persistent_object, target_rest_class)

    # would raise an exception if the attribute_filter differs from the target_rest_class
    if attribute_filter is not None and isinstance(attribute_filter, parser.AttributeFilter):
        parser.AttributeFilter.from_model(target_rest_class).conforms_to_template_filter(attribute_filter)

    # convert filter to immutable if it isn't already
    if isinstance(attribute_filter, parser.AttributeFilter):
        attribute_filter = attribute_filter.as_immutable()

    return adapter_instance.adapt_persistent_to_rest(persistent_object, attribute_filter)