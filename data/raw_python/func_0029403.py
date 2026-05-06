def convert_dotted(params):
        """ Convert dotted keys in :params: dictset to a nested dictset.

        E.g. {'settings.foo': 'bar'} -> {'settings': {'foo': 'bar'}}
        """
        if not isinstance(params, dictset):
            params = dictset(params)

        dotted_items = {k: v for k, v in params.items() if '.' in k}

        if dotted_items:
            dicts = [str2dict(key, val) for key, val in dotted_items.items()]
            dotted = six.functools.reduce(merge_dicts, dicts)
            params = params.subset(['-' + k for k in dotted_items.keys()])
            params.update(dict(dotted))

        return params