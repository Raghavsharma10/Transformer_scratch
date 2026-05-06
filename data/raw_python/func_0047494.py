def supports_currency_type(self, currency_type=None):
        """Tests if the given currency type is supported.

        arg:    currency_type (osid.type.Type): a currency Type
        return: (boolean) - ``true`` if the type is supported, ``false``
                otherwise
        raise:  IllegalState - syntax is not a ``CURRENCY``
        raise:  NullArgument - ``currency_type`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.Metadata.supports_coordinate_type
        from .osid_errors import IllegalState, NullArgument
        if not currency_type:
            raise NullArgument('no input Type provided')
        if self._kwargs['syntax'] not in ['``CURRENCY``']:
            raise IllegalState('put more meaninful message here')
        return currency_type in self.get_currency_types