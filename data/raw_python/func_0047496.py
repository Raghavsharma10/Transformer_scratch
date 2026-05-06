def supports_time_type(self, time_type=None):
        """Tests if the given time type is supported.

        arg:    time_type (osid.type.Type): a time Type
        return: (boolean) - ``true`` if the type is supported, ``false``
                otherwise
        raise:  IllegalState - syntax is not a ``DATETIME, DURATION,``
                or ``TIME``
        raise:  NullArgument - ``time_type`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.Metadata.supports_coordinate_type
        from .osid_errors import IllegalState, NullArgument
        if not time_type:
            raise NullArgument('no input Type provided')
        if self._kwargs['syntax'] not in ['``DATETIME,', 'DURATION,``', '``TIME``']:
            raise IllegalState('put more meaninful message here')
        return time_type in self.get_time_types