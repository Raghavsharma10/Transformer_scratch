def software_language(instance):
    """Ensure the 'language' property of software objects is a valid ISO 639-2
    language code.
    """
    for key, obj in instance['objects'].items():
        if ('type' in obj and obj['type'] == 'software' and
                'languages' in obj):
            for lang in obj['languages']:
                if lang not in enums.SOFTWARE_LANG_CODES:
                    yield JSONError("The 'languages' property of object '%s' "
                                    "contains an invalid ISO 639-2 language "
                                    " code ('%s')."
                                    % (key, lang), instance['id'])