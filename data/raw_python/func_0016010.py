def custom_object_prefix_lax(instance):
    """Ensure custom objects follow lenient naming style conventions
    for forward-compatibility.
    """
    if (instance['type'] not in enums.TYPES and
            instance['type'] not in enums.RESERVED_OBJECTS and
            not CUSTOM_TYPE_LAX_PREFIX_RE.match(instance['type'])):
        yield JSONError("Custom object type '%s' should start with 'x-' in "
                        "order to be compatible with future versions of the "
                        "STIX 2 specification." % instance['type'],
                        instance['id'], 'custom-prefix-lax')