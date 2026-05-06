def asdict(self, name, _type=None, _set=False):
        """
        Turn this 'a:2,b:blabla,c:True,a:'d' to
        {a:[2, 'd'], b:'blabla', c:True}

        """

        if _type is None:
            _type = lambda t: t

        dict_str = self.pop(name, None)
        if not dict_str:
            return {}

        _dict = {}
        for item in split_strip(dict_str):
            key, _, val = item.partition(':')
            val = _type(val)
            if key in _dict:
                if isinstance(_dict[key], list):
                    _dict[key].append(val)
                else:
                    _dict[key] = [_dict[key], val]
            else:
                _dict[key] = val

        if _set:
            self[name] = _dict

        return _dict