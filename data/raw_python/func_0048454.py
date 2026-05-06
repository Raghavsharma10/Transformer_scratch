def is_ancestor_of_catalog(self, id_, catalog_id):
        """Tests if an ``Id`` is an ancestor of a catalog.

        arg:    id (osid.id.Id): an ``Id``
        arg:    catalog_id (osid.id.Id): the ``Id`` of a catalog
        return: (boolean) - ``true`` if this ``id`` is an ancestor of
                ``catalogId``.  ``false`` otherwise
        raise:  NotFound - ``catalog_id`` not found
        raise:  NullArgument - ``catalog_id`` or ``id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*
        *implementation notes*: If ``id`` not found return ``false``.

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.is_ancestor_of_bin
        if self._catalog_session is not None:
            return self._catalog_session.is_ancestor_of_catalog(id_=id_, catalog_id=catalog_id)
        return self._hierarchy_session.is_ancestor(id_=id_, ancestor_id=catalog_id)