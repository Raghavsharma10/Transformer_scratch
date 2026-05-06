def _parse_query_key(self, key, val, is_escaped):
        """
        Strips query modifier from key and call's the appropriate value modifier.

        Args:
            key (str): Query key
            val: Query value

        Returns:
            Parsed query key and value.
        """
        if key.endswith('__contains'):
            key = key[:-10]
            val = self._parse_query_modifier('contains', val, is_escaped)
        elif key.endswith('__range'):
            key = key[:-7]
            val = self._parse_query_modifier('range', val, is_escaped)
        elif key.endswith('__startswith'):
            key = key[:-12]
            val = self._parse_query_modifier('startswith', val, is_escaped)
        elif key.endswith('__endswith'):
            key = key[:-10]
            val = self._parse_query_modifier('endswith', val, is_escaped)
        # lower than
        elif key.endswith('__lt'):
            key = key[:-4]
            val = self._parse_query_modifier('lt', val, is_escaped)
        # greater than
        elif key.endswith('__gt'):
            key = key[:-4]
            val = self._parse_query_modifier('gt', val, is_escaped)
        # lower than or equal
        elif key.endswith('__lte'):
            key = key[:-5]
            val = self._parse_query_modifier('lte', val, is_escaped)
        # greater than or equal
        elif key.endswith('__gte'):
            key = key[:-5]
            val = self._parse_query_modifier('gte', val, is_escaped)
        elif key != 'NOKEY' and not is_escaped:
            val = self._escape_query(val)
        return key, val