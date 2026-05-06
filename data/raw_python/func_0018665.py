def parse(key_string: str) -> 'Key':
        """ Parses a flat key string and returns a key """
        parts = key_string.split(Key.PARTITION)
        key_type = KeyType.DIMENSION
        if parts[3]:
            key_type = KeyType.TIMESTAMP
        return Key(key_type, parts[0], parts[1], parts[2].split(Key.DIMENSION_PARTITION)
                   if parts[2] else [],
                   parser.parse(parts[3]) if parts[3] else None)