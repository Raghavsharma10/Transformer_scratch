def can_create_bin_with_record_types(self, bin_record_types):
        """Tests if this user can create a single ``Bin`` using the desired record types.

        While ``ResourceManager.getBinRecordTypes()`` can be used to
        examine which records are supported, this method tests which
        record(s) are required for creating a specific ``Bin``.
        Providing an empty array tests if a ``Bin`` can be created with
        no records.

        arg:    bin_record_types (osid.type.Type[]): array of bin record
                types
        return: (boolean) - ``true`` if ``Bin`` creation using the
                specified ``Types`` is supported, ``false`` otherwise
        raise:  NullArgument - ``bin_record_types`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinAdminSession.can_create_bin_with_record_types
        # NOTE: It is expected that real authentication hints will be
        # handled in a service adapter above the pay grade of this impl.
        if self._catalog_session is not None:
            return self._catalog_session.can_create_catalog_with_record_types(catalog_record_types=bin_record_types)
        return True