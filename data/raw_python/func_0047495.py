def supports_calendar_type(self, calendar_type=None):
        """Tests if the given calendar type is supported.

        arg:    calendar_type (osid.type.Type): a calendar Type
        return: (boolean) - ``true`` if the type is supported, ``false``
                otherwise
        raise:  IllegalState - syntax is not a ``DATETIME`` or
                ``DURATION``
        raise:  NullArgument - ``calendar_type`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.Metadata.supports_coordinate_type
        from .osid_errors import IllegalState, NullArgument
        if not calendar_type:
            raise NullArgument('no input Type provided')
        if self._kwargs['syntax'] not in ['``DATETIME``', '``DURATION``']:
            raise IllegalState('put more meaninful message here')
        return calendar_type in self.get_calendar_types