def is_parent_of_bank(self, id_, bank_id):
        """Tests if an ``Id`` is a direct parent of a bank.

        arg:    id (osid.id.Id): an ``Id``
        arg:    bank_id (osid.id.Id): the ``Id`` of a bank
        return: (boolean) - ``true`` if this ``id`` is a parent of
                ``bank_id,``  ``false`` otherwise
        raise:  NotFound - ``bank_id`` is not found
        raise:  NullArgument - ``id`` or ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*
        *implementation notes*: If ``id`` not found return ``false``.

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.is_parent_of_bin
        if self._catalog_session is not None:
            return self._catalog_session.is_parent_of_catalog(id_=id_, catalog_id=bank_id)
        return self._hierarchy_session.is_parent(id_=bank_id, parent_id=id_)