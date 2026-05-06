def supports_time_type(self, time_type):
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
        if self._kwargs['syntax'] not in ['``DATETIME,', 'DURATION,``', '``TIME``']:
            raise errors.IllegalState()
        return time_type in self.get_time_types