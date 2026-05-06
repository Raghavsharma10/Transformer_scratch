def get_osid_temporal_mdata():
    """Return default mdata map for OsidTemporal"""
    return {
        'start_date': {
            'element_label': {
                'text': 'Start Date',
                'languageTypeId': str(DEFAULT_LANGUAGE_TYPE),
                'scriptTypeId': str(DEFAULT_SCRIPT_TYPE),
                'formatTypeId': str(DEFAULT_FORMAT_TYPE),
            },
            'instructions': {
                'text': 'enter a valid datetime object.',
                'languageTypeId': str(DEFAULT_LANGUAGE_TYPE),
                'scriptTypeId': str(DEFAULT_SCRIPT_TYPE),
                'formatTypeId': str(DEFAULT_FORMAT_TYPE),
            },
            'required': True,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_date_time_values': [datetime.datetime.min],
            'syntax': 'DATETIME',
            'date_time_set': [],
        },
        'end_date': {
            'element_label': {
                'text': 'Start Date',
                'languageTypeId': str(DEFAULT_LANGUAGE_TYPE),
                'scriptTypeId': str(DEFAULT_SCRIPT_TYPE),
                'formatTypeId': str(DEFAULT_FORMAT_TYPE),
            },
            'instructions': {
                'text': 'enter a valid datetime object.',
                'languageTypeId': str(DEFAULT_LANGUAGE_TYPE),
                'scriptTypeId': str(DEFAULT_SCRIPT_TYPE),
                'formatTypeId': str(DEFAULT_FORMAT_TYPE),
            },
            'required': True,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_date_time_values': [datetime.datetime.max],
            'syntax': 'DATETIME',
            'date_time_set': [],
        }
    }