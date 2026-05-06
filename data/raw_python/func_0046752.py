def alias_gradebook(self, gradebook_id, alias_id):
        """Adds an ``Id`` to a ``Gradebook`` for the purpose of creating compatibility.

        The primary ``Id`` of the ``Gradebook`` is determined by the
        provider. The new ``Id`` performs as an alias to the primary
        ``Id`` . If the alias is a pointer to another gradebook, it is
        reassigned to the given gradebook ``Id``.

        arg:    gradebook_id (osid.id.Id): the ``Id`` of a ``Gradebook``
        arg:    alias_id (osid.id.Id): the alias ``Id``
        raise:  AlreadyExists - ``alias_id`` is already assigned
        raise:  NotFound - ``gradebook_id`` not found
        raise:  NullArgument - ``gradebook_id`` or ``alias_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinLookupSession.alias_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.alias_catalog(catalog_id=gradebook_id, alias_id=alias_id)
        self._alias_id(primary_id=gradebook_id, equivalent_id=alias_id)