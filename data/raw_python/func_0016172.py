def make_avsc_object(json_data, names=None):
    # type: (Union[Dict[Text, Text], List[Any], Text], Optional[Names]) -> Schema
    """
    Build Avro Schema from data parsed out of JSON string.

    @arg names: A Name object (tracks seen names and default space)
    """
    if names is None:
        names = Names()
    assert isinstance(names, Names)

    # JSON object (non-union)
    if hasattr(json_data, 'get') and callable(json_data.get):  # type: ignore
        assert isinstance(json_data, Dict)
        atype = cast(Text, json_data.get('type'))
        other_props = get_other_props(json_data, SCHEMA_RESERVED_PROPS)
        if atype in PRIMITIVE_TYPES:
            return PrimitiveSchema(atype, other_props)
        if atype in NAMED_TYPES:
            name = cast(Text, json_data.get('name'))
            namespace = cast(Text, json_data.get('namespace',
                                                 names.default_namespace))
            if atype == 'enum':
                symbols = cast(List[Text], json_data.get('symbols'))
                doc = json_data.get('doc')
                return EnumSchema(name, namespace, symbols, names, doc, other_props)
            if atype in ['record', 'error']:
                fields = cast(List, json_data.get('fields'))
                doc = json_data.get('doc')
                return RecordSchema(name, namespace, fields, names, atype, doc, other_props)
            raise SchemaParseException('Unknown Named Type: %s' % atype)
        if atype in VALID_TYPES:
            if atype == 'array':
                items = cast(List, json_data.get('items'))
                return ArraySchema(items, names, other_props)
        if atype is None:
            raise SchemaParseException('No "type" property: %s' % json_data)
        raise SchemaParseException('Undefined type: %s' % atype)
    # JSON array (union)
    if isinstance(json_data, list):
        return UnionSchema(json_data, names)
    # JSON string (primitive)
    if json_data in PRIMITIVE_TYPES:
        return PrimitiveSchema(cast(Text, json_data))
    # not for us!
    fail_msg = "Could not make an Avro Schema object from %s." % json_data
    raise SchemaParseException(fail_msg)