def get_osid_containable_mdata():
    """Return default mdata map for OsidContainable"""
    return {
        'sequestered': {
            'element_label': {
                'text': 'sequestered',
                'languageTypeId': str(DEFAULT_LANGUAGE_TYPE),
                'scriptTypeId': str(DEFAULT_SCRIPT_TYPE),
                'formatTypeId': str(DEFAULT_FORMAT_TYPE),
            },
            'instructions': {
                'text': 'enter either true or false.',
                'languageTypeId': str(DEFAULT_LANGUAGE_TYPE),
                'scriptTypeId': str(DEFAULT_SCRIPT_TYPE),
                'formatTypeId': str(DEFAULT_FORMAT_TYPE),
            },
            'required': False,
            'read_only': False,
            'linked': False,
            'default_boolean_values': [False],
            'array': False,
            'syntax': 'BOOLEAN',
        }
    }