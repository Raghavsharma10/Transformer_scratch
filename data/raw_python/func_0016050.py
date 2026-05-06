def properties_strict(instance):
    """Ensure that no custom properties are used, but only the official ones
    from the specification.
    """
    if instance['type'] not in enums.TYPES:
        return  # only check properties for official objects

    defined_props = enums.PROPERTIES.get(instance['type'], [])
    for prop in instance.keys():
        if prop not in defined_props:
            yield JSONError("Property '%s' is not one of those defined in the"
                            " specification." % prop, instance['id'])

    if has_cyber_observable_data(instance):
        for key, obj in instance['objects'].items():
            type_ = obj.get('type', '')
            if type_ not in enums.OBSERVABLE_PROPERTIES:
                continue  # custom observable types handled outside this function
            observable_props = enums.OBSERVABLE_PROPERTIES.get(type_, [])
            embedded_props = enums.OBSERVABLE_EMBEDDED_PROPERTIES.get(type_, {})
            extensions = enums.OBSERVABLE_EXTENSIONS.get(type_, [])
            for prop in obj.keys():
                if prop not in observable_props:
                    yield JSONError("Property '%s' is not one of those defined in the"
                                    " specification for %s objects."
                                    % (prop, type_), instance['id'])
                # Check properties of embedded cyber observable types
                elif prop in embedded_props:
                    embedded_prop_keys = embedded_props.get(prop, [])
                    for embedded_key in obj[prop]:
                        if isinstance(embedded_key, dict):
                            for embedded in embedded_key:
                                if embedded not in embedded_prop_keys:
                                    yield JSONError("Property '%s' is not one of those defined in the"
                                                    " specification for the %s property in %s objects."
                                                    % (embedded, prop, type_), instance['id'])
                        elif embedded_key not in embedded_prop_keys:
                            yield JSONError("Property '%s' is not one of those defined in the"
                                            " specification for the %s property in %s objects."
                                            % (embedded_key, prop, type_), instance['id'])

            # Check properties of embedded cyber observable types
            for ext_key in obj.get('extensions', {}):
                if ext_key not in extensions:
                    continue  # don't check custom extensions
                extension_props = enums.OBSERVABLE_EXTENSION_PROPERTIES[ext_key]
                for ext_prop in obj['extensions'][ext_key]:
                    if ext_prop not in extension_props:
                        yield JSONError("Property '%s' is not one of those defined in the"
                                        " specification for the %s extension in %s objects."
                                        % (ext_prop, ext_key, type_), instance['id'])
                    embedded_ext_props = enums.OBSERVABLE_EXTENSION_EMBEDDED_PROPERTIES.get(ext_key, {}).get(ext_prop, [])
                    if embedded_ext_props:
                        for embed_ext_prop in obj['extensions'][ext_key].get(ext_prop, []):
                            if embed_ext_prop not in embedded_ext_props:
                                yield JSONError("Property '%s' in the %s property of the %s extension "
                                                "is not one of those defined in the specification."
                                                % (embed_ext_prop, ext_prop, ext_key), instance['id'])