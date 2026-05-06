def _encode(self, obj):
        """Returns a JSON representation of a Python object - see dumps.
        Accepts objects of any type, calls the appropriate type-specific encoder.
        """

        if self._use_hook:
            obj = self.encode_hook(obj)

        # first try simple strict checks

        _objtype = obj.__class__

        if _objtype is str:
            return self._encode_str(obj)

        if _objtype is bool:
            if obj:
                return 'true'
            else:
                return 'false'

        if _objtype is int or _objtype is float:
            return self._encode_numbers(obj)

        if _objtype is list or _objtype is tuple:
            return self._encode_list(obj)

        if obj is None:
            return 'null'

        if _objtype is dict or obj is OrderedDict:
            return self._encode_dict(obj)

        if _objtype is UUID:
            return '"' + str(obj) + '"'

        if _objtype is Decimal:
            return '"' + str(obj) + '"'

        # For all non-std types try __mm_json__ and then __mm_serialize__ before any isinstance
        # checks

        try:
            sx_json_data = obj.__mm_json__
        except AttributeError:
            pass
        else:
            try:
                data = sx_json_data()
            except NotImplementedError:
                pass
            else:
                if isinstance(data, bytes):
                    return data.decode('utf-8')
                else:
                    return self._encode_str(data, escape_quotes=False)
        try:
            sx_encoder = obj.__mm_serialize__
        except AttributeError:
            pass
        else:
            try:
                data = sx_encoder()
            except NotImplementedError:
                pass
            else:
                return self._encode(data)

        # do more in-depth class analysis

        if isinstance(obj, UUID):
            return '"' + str(obj) + '"'

        if isinstance(obj, str):
            return self._encode_str(obj)

        if isinstance(obj, (list, tuple, set, frozenset, Set)):
            return self._encode_list(obj)

        if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray)):
            return self._encode_list(obj)

        if isinstance(obj, (dict, OrderedDict, Mapping)):
            return self._encode_dict(obj)

        # note: number checks using isinstance should come after True/False checks
        if isinstance(obj, Number):
            return self._encode_numbers(obj)

        if isinstance(obj, (date, time)):
            return '"' + obj.isoformat() + '"'

        return self._encode(self.default(obj))