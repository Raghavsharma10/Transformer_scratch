def supports_string_match_type(self, string_match_type):
        """Tests if the given string match type is supported.

        arg:    string_match_type (osid.type.Type): a string match type
        return: (boolean) - ``true`` if the given string match type Is
                supported, ``false`` otherwise
        raise:  IllegalState - syntax is not a ``STRING``
        raise:  NullArgument - ``string_match_type`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.Metadata.supports_coordinate_type
        if self._kwargs['syntax'] not in ['``STRING``']:
            raise errors.IllegalState()
        return string_match_type in self.get_string_match_types