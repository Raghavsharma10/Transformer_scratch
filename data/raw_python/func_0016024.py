def custom_observable_object_prefix_lax(instance):
    """Ensure custom observable objects follow naming style conventions.
    """
    for key, obj in instance['objects'].items():
        if ('type' in obj and obj['type'] not in enums.OBSERVABLE_TYPES and
                obj['type'] not in enums.OBSERVABLE_RESERVED_OBJECTS and
                not CUSTOM_TYPE_LAX_PREFIX_RE.match(obj['type'])):
            yield JSONError("Custom Observable Object type '%s' should start "
                            "with 'x-'."
                            % obj['type'], instance['id'],
                            'custom-prefix-lax')