def supports_object_type(self, object_type=None):
        """Tests if the given object type is supported.

        arg:    object_type (osid.type.Type): an object Type
        return: (boolean) - ``true`` if the type is supported, ``false``
                otherwise
        raise:  IllegalState - syntax is not an ``OBJECT``
        raise:  NullArgument - ``object_type`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.Metadata.supports_coordinate_type
        from .osid_errors import IllegalState, NullArgument
        if not object_type:
            raise NullArgument('no input Type provided')
        if self._kwargs['syntax'] not in ['``OBJECT``']:
            raise IllegalState('put more meaninful message here')
        return object_type in self.get_object_types