def custom_object_prefix_strict(instance):
    """Ensure custom objects follow strict naming style conventions.
    """
    if (instance['type'] not in enums.TYPES and
            instance['type'] not in enums.RESERVED_OBJECTS and
            not CUSTOM_TYPE_PREFIX_RE.match(instance['type'])):
        yield JSONError("Custom object type '%s' should start with 'x-' "
                        "followed by a source unique identifier (like a "
                        "domain name with dots replaced by hyphens), a hyphen "
                        "and then the name." % instance['type'],
                        instance['id'], 'custom-prefix')