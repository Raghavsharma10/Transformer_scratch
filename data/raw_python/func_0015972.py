def getOSExtPropDesc(data_type, name, value):
    """
    Returns an OS extension property descriptor.
    data_type (int)
        See wPropertyDataType documentation.
    name (string)
        See PropertyName documentation.
    value (string)
        See PropertyData documentation.
        NULL chars must be explicitely included in the value when needed,
        this function does not add any terminating NULL for example.
    """
    klass = type(
        'OSExtPropDesc',
        (OSExtPropDescHead, ),
        {
            '_fields_': [
                ('bPropertyName', ctypes.c_char * len(name)),
                ('dwPropertyDataLength', le32),
                ('bProperty', ctypes.c_char * len(value)),
            ],
        }
    )
    return klass(
        dwSize=ctypes.sizeof(klass),
        dwPropertyDataType=data_type,
        wPropertyNameLength=len(name),
        bPropertyName=name,
        dwPropertyDataLength=len(value),
        bProperty=value,
    )