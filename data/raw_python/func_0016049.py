def types_strict(instance):
    """Ensure that no custom object types are used, but only the official ones
    from the specification.
    """
    if instance['type'] not in enums.TYPES:
        yield JSONError("Object type '%s' is not one of those defined in the"
                        " specification." % instance['type'], instance['id'])

    if has_cyber_observable_data(instance):
        for key, obj in instance['objects'].items():
            if 'type' in obj and obj['type'] not in enums.OBSERVABLE_TYPES:
                yield JSONError("Observable object %s is type '%s' which is "
                                "not one of those defined in the "
                                "specification."
                                % (key, obj['type']), instance['id'])