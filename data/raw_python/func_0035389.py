def p_identifier_name_string(self, p):
        """identifier_name_string : identifier_name
        """
        p[0] = asttypes.PropIdentifier(p[1].value)
        # manually clone the position attributes.
        for k in ('_token_map', 'lexpos', 'lineno', 'colno'):
            setattr(p[0], k, getattr(p[1], k))