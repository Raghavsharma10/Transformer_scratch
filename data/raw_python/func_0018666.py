def parse_sort_key(identity: str, sort_key_string: str) -> 'Key':
        """ Parses a flat key string and returns a key """
        parts = sort_key_string.split(Key.PARTITION)
        key_type = KeyType.DIMENSION
        if parts[2]:
            key_type = KeyType.TIMESTAMP
        return Key(key_type, identity, parts[0], parts[1].split(Key.DIMENSION_PARTITION)
                   if parts[1] else [],
                   parser.parse(parts[2]) if parts[2] else None)