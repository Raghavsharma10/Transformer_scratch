def custom_prefix_strict(instance):
    """Ensure custom content follows strict naming style conventions.
    """
    for error in chain(custom_object_prefix_strict(instance),
                       custom_property_prefix_strict(instance),
                       custom_observable_object_prefix_strict(instance),
                       custom_object_extension_prefix_strict(instance),
                       custom_observable_properties_prefix_strict(instance)):
        yield error