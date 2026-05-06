def supports_coordinate_type(self, coordinate_type=None):
        """Tests if the given coordinate type is supported.

        arg:    coordinate_type (osid.type.Type): a coordinate Type
        return: (boolean) - ``true`` if the type is supported, ``false``
                otherwise
        raise:  IllegalState - syntax is not a ``COORDINATE``
        raise:  NullArgument - ``coordinate_type`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.Metadata.supports_coordinate_type
        from .osid_errors import IllegalState, NullArgument
        if not coordinate_type:
            raise NullArgument('no input Type provided')
        if self._kwargs['syntax'] not in ['``COORDINATE``']:
            raise IllegalState('put more meaninful message here')
        return coordinate_type in self.get_coordinate_types