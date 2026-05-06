def alias_bank(self, bank_id, alias_id):
        """Adds an ``Id`` to a ``Bank`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``Bank`` is determined by the
        provider. The new ``Id`` is an alias to the primary ``Id``. If
        the alias is a pointer to another bank, it is reassigned to the
        given bank ``Id``.

        arg:    bank_id (osid.id.Id): the ``Id`` of a ``Bank``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is in use as a primary
                ``Id``
        raise:  NotFound - ``bank_id`` not found
        raise:  NullArgument - ``bank_id`` or ``alias_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinLookupSession.alias_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.alias_catalog(catalog_id=bank_id, alias_id=alias_id)
        self._alias_id(primary_id=bank_id, equivalent_id=alias_id)