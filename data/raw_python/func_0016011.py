def custom_property_prefix_strict(instance):
    """Ensure custom properties follow strict naming style conventions.

    Does not check property names in custom objects.
    """
    for prop_name in instance.keys():
        if (instance['type'] in enums.PROPERTIES and
                prop_name not in enums.PROPERTIES[instance['type']] and
                prop_name not in enums.RESERVED_PROPERTIES and
                not CUSTOM_PROPERTY_PREFIX_RE.match(prop_name)):

            yield JSONError("Custom property '%s' should have a type that "
                            "starts with 'x_' followed by a source unique "
                            "identifier (like a domain name with dots "
                            "replaced by hyphen), a hyphen and then the name."
                            % prop_name, instance['id'],
                            'custom-prefix')