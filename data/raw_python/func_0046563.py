def alias_family(self, family_id, alias_id):
        """Adds an ``Id`` to a ``Family`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``Family`` is determined by the
        provider. The new ``Id`` performs as an alias to the primary
        ``Id``. If the alias is a pointer to another family, it is
        reassigned to the given family ``Id``.

        arg:    family_id (osid.id.Id): the ``Id`` of a ``Family``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is already assigned
        raise:  NotFound - ``family_id`` not found
        raise:  NullArgument - ``family_id`` or ``alias_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinLookupSession.alias_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.alias_catalog(catalog_id=family_id, alias_id=alias_id)
        self._alias_id(primary_id=family_id, equivalent_id=alias_id)