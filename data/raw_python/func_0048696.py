def supports_spatial_unit_record_type(self, spatial_unit_record_type):
        """Tests if the given spatial unit record type is supported.

        arg:    spatial_unit_record_type (osid.type.Type): a spatial
                unit record Type
        return: (boolean) - ``true`` if the type is supported, ``false``
                otherwise
        raise:  IllegalState - syntax is not an ``SPATIALUNIT``
        raise:  NullArgument - ``spatial_unit_record_type`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.Metadata.supports_coordinate_type
        if self._kwargs['syntax'] not in ['``SPATIALUNIT``']:
            raise errors.IllegalState()
        return spatial_unit_record_type in self.get_spatial_unit_record_types