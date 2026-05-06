def supports_heading_type(self, heading_type):
        """Tests if the given heading type is supported.

        arg:    heading_type (osid.type.Type): a heading Type
        return: (boolean) - ``true`` if the type is supported, ``false``
                otherwise
        raise:  IllegalState - syntax is not a ``HEADING``
        raise:  NullArgument - ``heading_type`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.Metadata.supports_coordinate_type
        if self._kwargs['syntax'] not in ['``HEADING``']:
            raise errors.IllegalState()
        return heading_type in self.get_heading_types