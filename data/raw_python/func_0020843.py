def flatten_nested_hash(hash_table):
    """
    Flatten nested dictionary for GET / POST / DELETE API request
    """
    def flatten(hash_table, brackets=True):
        f = {}
        for key, value in hash_table.items():
            _key = '[' + str(key) + ']' if brackets else str(key)
            if isinstance(value, dict):
                for k, v in flatten(value).items():
                    f[_key + k] = v
            elif isinstance(value, list):
                temp_hash = {}
                for i, v in enumerate(value):
                    temp_hash[str(i)] = v
                for k, v in flatten(temp_hash).items():
                    f[_key + k] = v
            else:
                f[_key] = value
        return f
    return flatten(hash_table, False)