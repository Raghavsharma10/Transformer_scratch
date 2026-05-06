def type_last(self, obj: JsonObj) -> JsonObj:
        """ Move the type identifiers to the end of the object for print purposes """
        def _tl_list(v: List) -> List:
            return [self.type_last(e) if isinstance(e, JsonObj)
                                   else _tl_list(e) if isinstance(e, list) else e for e in v if e is not None]

        rval = JsonObj()
        for k in as_dict(obj).keys():
            v = obj[k]
            if v is not None and k not in ('type', '_context'):
                rval[k] = _tl_list(v) if isinstance(v, list) else self.type_last(v) if isinstance(v, JsonObj) else v

        if 'type' in obj and obj.type:
            rval.type = obj.type
        return rval