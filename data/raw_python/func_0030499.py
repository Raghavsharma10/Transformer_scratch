def _as_document(self, identifier):
        """ Converts given identifier to the document indexed by FTS backend.

        Args:
            identifier (dict): identifier to convert. Dict contains at
                least 'identifier', 'type' and 'name' keys.

        Returns:
            dict with structure matches to BaseIdentifierIndex._schema.

        """
        return {
            'identifier': u('{}').format(identifier['identifier']),
            'type': u('{}').format(identifier['type']),
            'name': u('{}').format(identifier['name'])
        }