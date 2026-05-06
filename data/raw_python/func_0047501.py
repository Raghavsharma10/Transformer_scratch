def supports_version_type(self, version_type=None):
        """Tests if the given version type is supported.

        arg:    version_type (osid.type.Type): a version Type
        return: (boolean) - ``true`` if the type is supported, ``false``
                otherwise
        raise:  IllegalState - syntax is not a ``VERSION``
        raise:  NullArgument - ``version_type`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.Metadata.supports_coordinate_type
        from .osid_errors import IllegalState, NullArgument
        if not version_type:
            raise NullArgument('no input Type provided')
        if self._kwargs['syntax'] not in ['``VERSION``']:
            raise IllegalState('put more meaninful message here')
        return version_type in self.get_version_types