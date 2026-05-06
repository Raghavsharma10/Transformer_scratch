def custom_object_extension_prefix_lax(instance):
    """Ensure custom observable object extensions follow naming style
    conventions.
    """
    for key, obj in instance['objects'].items():
        if not ('extensions' in obj and 'type' in obj and
                obj['type'] in enums.OBSERVABLE_EXTENSIONS):
            continue
        for ext_key in obj['extensions']:
            if (ext_key not in enums.OBSERVABLE_EXTENSIONS[obj['type']] and
                    not CUSTOM_TYPE_LAX_PREFIX_RE.match(ext_key)):
                yield JSONError("Custom Cyber Observable Object extension type"
                                " '%s' should start with 'x-'."
                                % ext_key, instance['id'],
                                'custom-prefix-lax')