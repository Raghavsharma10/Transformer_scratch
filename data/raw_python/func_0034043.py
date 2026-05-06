def parse_address(self, address_line):
        """
        Parses the given address into it's individual address fields.
        """
        params = {"term": address_line}
        json = self._make_request('/address/getParsedAddress', params)
        if json is None:
            return None
        return Address.from_json(json)