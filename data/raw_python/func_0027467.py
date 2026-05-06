def _serializeBooleans(params):
        """"Convert all booleans to lowercase strings"""
        serialized = {}
        for name, value in params.items():
            if value is True:
                value = 'true'
            elif value is False:
                value = 'false'
            serialized[name] = value
        return serialized

        for k, v in params.items():
            if isinstance(v, bool):
                params[k] = str(v).lower()