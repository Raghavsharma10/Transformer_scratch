def alias_hierarchy(self, hierarchy_id, alias_id):
        """Adds an ``Id`` to a ``Hierarchy`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``Hierarchy`` is determined by the
        provider. The new ``Id`` performs as an alias to the primary
        ``Id``. If the alias is a pointer to another vault it is
        reassigned to the given vault ``Id``.

        arg:    hierarchy_id (osid.id.Id): the ``Id`` of an
                ``Hierarchy``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is already assigned
        raise:  NotFound - ``hierarchy_id`` not found
        raise:  NullArgument - ``hierarchy_id`` or ``alias_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinLookupSession.alias_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.alias_catalog(catalog_id=hierarchy_id, alias_id=alias_id)
        self._alias_id(primary_id=hierarchy_id, equivalent_id=alias_id)