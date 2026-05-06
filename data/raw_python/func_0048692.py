def supports_calendar_type(self, calendar_type):
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
        if self._kwargs['syntax'] not in ['``DATETIME``', '``DURATION``']:
            raise errors.IllegalState()
        return calendar_type in self.get_calendar_types