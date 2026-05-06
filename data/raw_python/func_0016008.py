def custom_prefix_lax(instance):
    """Ensure custom content follows lenient naming style conventions
    for forward-compatibility.
    """
    for error in chain(custom_object_prefix_lax(instance),
                       custom_property_prefix_lax(instance),
                       custom_observable_object_prefix_lax(instance),
                       custom_object_extension_prefix_lax(instance),
                       custom_observable_properties_prefix_lax(instance)):
        yield error