def alias_bin(self, bin_id, alias_id):
        """Adds an ``Id`` to a ``Bin`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``Bin`` is determined by the provider.
        The new ``Id`` performs as an alias to the primary ``Id``. If
        the alias is a pointer to another bin, it is reassigned to the
        given bin ``Id``.

        arg:    bin_id (osid.id.Id): the ``Id`` of a ``Bin``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is already assigned
        raise:  NotFound - ``bin_id`` not found
        raise:  NullArgument - ``bin_id`` or ``alias_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinLookupSession.alias_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.alias_catalog(catalog_id=bin_id, alias_id=alias_id)
        self._alias_id(primary_id=bin_id, equivalent_id=alias_id)