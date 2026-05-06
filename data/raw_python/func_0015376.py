def map_object_literal(v: Union[str, jsonasobj.JsonObj]) -> ShExJ.ObjectLiteral:
    """ `PyShEx.jsg <https://github.com/hsolbrig/ShExJSG/ShExJSG/ShExJ.jsg>`_ does not add identifying
    types to ObjectLiterals.  This routine re-identifies the types
    """
    # TODO: isinstance(v, JSGString) should work here, but it doesn't with IRIREF(http://a.example/v1)
    return v if issubclass(type(v), JSGString) or (isinstance(v, JSGObject) and 'type' in v) else \
        ShExJ.IRIREF(v) if isinstance(v, str) else ShExJ.ObjectLiteral(**v._as_dict)