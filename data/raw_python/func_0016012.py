def custom_property_prefix_lax(instance):
    """Ensure custom properties follow lenient naming style conventions
    for forward-compatibility.

    Does not check property names in custom objects.
    """
    for prop_name in instance.keys():
        if (instance['type'] in enums.PROPERTIES and
                prop_name not in enums.PROPERTIES[instance['type']] and
                prop_name not in enums.RESERVED_PROPERTIES and
                not CUSTOM_PROPERTY_LAX_PREFIX_RE.match(prop_name)):

            yield JSONError("Custom property '%s' should have a type that "
                            "starts with 'x_' in order to be compatible with "
                            "future versions of the STIX 2 specification." %
                            prop_name, instance['id'],
                            'custom-prefix-lax')