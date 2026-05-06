def custom_observable_object_prefix_strict(instance):
    """Ensure custom observable objects follow strict naming style conventions.
    """
    for key, obj in instance['objects'].items():
        if ('type' in obj and obj['type'] not in enums.OBSERVABLE_TYPES and
                obj['type'] not in enums.OBSERVABLE_RESERVED_OBJECTS and
                not CUSTOM_TYPE_PREFIX_RE.match(obj['type'])):
            yield JSONError("Custom Observable Object type '%s' should start "
                            "with 'x-' followed by a source unique identifier "
                            "(like a domain name with dots replaced by "
                            "hyphens), a hyphen and then the name."
                            % obj['type'], instance['id'],
                            'custom-prefix')