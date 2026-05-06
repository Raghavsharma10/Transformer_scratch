def cpp_spec():
    """C++ specification, provided for example, and java compatible."""
    return {
        INDENTATION    : '\t',
        BEG_BLOCK      : '{',
        END_BLOCK      : '}',
        BEG_LINE       : '',
        END_LINE       : '\n',
        BEG_ACTION     : '',
        END_ACTION     : ';',
        BEG_CONDITION  : 'if(',
        END_CONDITION  : ')',
        LOGICAL_AND    : ' && ',
        LOGICAL_OR     : ' || '
    }