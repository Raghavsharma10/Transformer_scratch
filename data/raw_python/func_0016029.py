def mime_type(instance):
    """Ensure the 'mime_type' property of file objects comes from the Template
    column in the IANA media type registry.
    """
    mime_pattern = re.compile(r'^(application|audio|font|image|message|model'
                              '|multipart|text|video)/[a-zA-Z0-9.+_-]+')
    for key, obj in instance['objects'].items():
        if ('type' in obj and obj['type'] == 'file' and 'mime_type' in obj):
            if enums.media_types():
                if obj['mime_type'] not in enums.media_types():
                    yield JSONError("The 'mime_type' property of object '%s' "
                                    "('%s') should be an IANA registered MIME "
                                    "Type of the form 'type/subtype'."
                                    % (key, obj['mime_type']), instance['id'],
                                    'mime-type')
            else:
                info("Can't reach IANA website; using regex for mime types.")
                if not mime_pattern.match(obj['mime_type']):
                    yield JSONError("The 'mime_type' property of object '%s' "
                                    "('%s') should be an IANA MIME Type of the"
                                    " form 'type/subtype'."
                                    % (key, obj['mime_type']), instance['id'],
                                    'mime-type')