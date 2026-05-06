def custom_observable_properties_prefix_strict(instance):
    """Ensure observable object custom properties follow strict naming style
    conventions.
    """
    for key, obj in instance['objects'].items():
        if 'type' not in obj:
            continue
        type_ = obj['type']

        for prop in obj:
            # Check objects' properties
            if (type_ in enums.OBSERVABLE_PROPERTIES and
                prop not in enums.OBSERVABLE_PROPERTIES[type_] and
                    not CUSTOM_PROPERTY_PREFIX_RE.match(prop)):
                yield JSONError("Cyber Observable Object custom property '%s' "
                                "should start with 'x_' followed by a source "
                                "unique identifier (like a domain name with "
                                "dots replaced by hyphens), a hyphen and then the"
                                " name."
                                % prop, instance['id'],
                                'custom-prefix')
            # Check properties of embedded cyber observable types
            if (type_ in enums.OBSERVABLE_EMBEDDED_PROPERTIES and
                    prop in enums.OBSERVABLE_EMBEDDED_PROPERTIES[type_]):
                for embed_prop in obj[prop]:
                    if isinstance(embed_prop, dict):
                        for embedded in embed_prop:
                            if (embedded not in enums.OBSERVABLE_EMBEDDED_PROPERTIES[type_][prop] and
                                    not CUSTOM_PROPERTY_PREFIX_RE.match(embedded)):
                                yield JSONError("Cyber Observable Object custom "
                                                "property '%s' in the %s property of "
                                                "%s object should start with 'x_' "
                                                "followed by a source unique "
                                                "identifier (like a domain name with "
                                                "dots replaced by hyphens), a hyphen and "
                                                "then the name."
                                                % (embedded, prop, type_), instance['id'],
                                                'custom-prefix')
                    elif (embed_prop not in enums.OBSERVABLE_EMBEDDED_PROPERTIES[type_][prop] and
                            not CUSTOM_PROPERTY_PREFIX_RE.match(embed_prop)):
                        yield JSONError("Cyber Observable Object custom "
                                        "property '%s' in the %s property of "
                                        "%s object should start with 'x_' "
                                        "followed by a source unique "
                                        "identifier (like a domain name with "
                                        "dots replaced by hyphens), a hyphen and "
                                        "then the name."
                                        % (embed_prop, prop, type_), instance['id'],
                                        'custom-prefix')

        # Check object extensions' properties
        if (type_ in enums.OBSERVABLE_EXTENSIONS and 'extensions' in obj):
            for ext_key in obj['extensions']:

                if ext_key in enums.OBSERVABLE_EXTENSIONS[type_]:
                    for ext_prop in obj['extensions'][ext_key]:
                        if (ext_prop not in enums.OBSERVABLE_EXTENSION_PROPERTIES[ext_key] and
                                not CUSTOM_PROPERTY_PREFIX_RE.match(ext_prop)):
                            yield JSONError("Cyber Observable Object custom "
                                            "property '%s' in the %s extension "
                                            "should start with 'x_' followed by a "
                                            "source unique identifier (like a "
                                            "domain name with dots replaced by "
                                            "hyphens), a hyphen and then the name."
                                            % (ext_prop, ext_key), instance['id'],
                                            'custom-prefix')

                if ext_key in enums.OBSERVABLE_EXTENSIONS[type_]:
                    for ext_prop in obj['extensions'][ext_key]:
                        if (ext_key in enums.OBSERVABLE_EXTENSION_EMBEDDED_PROPERTIES and
                                ext_prop in enums.OBSERVABLE_EXTENSION_EMBEDDED_PROPERTIES[ext_key]):
                            for embed_prop in obj['extensions'][ext_key][ext_prop]:
                                if not (isinstance(embed_prop, Iterable) and not isinstance(embed_prop, string_types)):
                                    embed_prop = [embed_prop]
                                for p in embed_prop:
                                    if (p not in enums.OBSERVABLE_EXTENSION_EMBEDDED_PROPERTIES[ext_key][ext_prop] and
                                            not CUSTOM_PROPERTY_PREFIX_RE.match(p)):
                                        yield JSONError("Cyber Observable Object "
                                                        "custom property '%s' in the %s "
                                                        "property of the %s extension should "
                                                        "start with 'x_' followed by a source "
                                                        "unique identifier (like a domain name"
                                                        " with dots replaced by hyphens), a "
                                                        "hyphen and then the name."
                                                        % (p, ext_prop, ext_key), instance['id'],
                                                        'custom-prefix')