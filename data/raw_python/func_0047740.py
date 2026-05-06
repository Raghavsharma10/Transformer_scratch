def get_gradebook_column_summary_mdata():
    """Return default mdata map for GradebookColumnSummary"""
    return {
        'gradebook_column': {
            'element_label': {
                'text': 'gradebook column',
                'languageTypeId': str(DEFAULT_LANGUAGE_TYPE),
                'scriptTypeId': str(DEFAULT_SCRIPT_TYPE),
                'formatTypeId': str(DEFAULT_FORMAT_TYPE),
            },
            'instructions': {
                'text': 'accepts an osid.id.Id object',
                'languageTypeId': str(DEFAULT_LANGUAGE_TYPE),
                'scriptTypeId': str(DEFAULT_SCRIPT_TYPE),
                'formatTypeId': str(DEFAULT_FORMAT_TYPE),
            },
            'required': False,
            'read_only': False,
            'linked': False,
            'array': False,
            'default_id_values': [''],
            'syntax': 'ID',
            'id_set': [],
        },
    }