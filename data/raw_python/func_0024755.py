def parse_raw(self, raw):
        """Parse alias array from raw bytes."""
        if not isinstance(raw, bytes):
            raise PyVLXException("AliasArray::invalid_type_if_raw", type_raw=type(raw))
        if len(raw) != 21:
            raise PyVLXException("AliasArray::invalid_size", size=len(raw))
        nbr_of_alias = raw[0]
        if nbr_of_alias > 5:
            raise PyVLXException("AliasArray::invalid_nbr_of_alias", nbr_of_alias=nbr_of_alias)
        for i in range(0, nbr_of_alias):
            self.alias_array_.append((raw[i*4+1:i*4+3], raw[i*4+3:i*4+5]))