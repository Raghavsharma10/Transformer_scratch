def is_descendant_of_catalog(self, id_, catalog_id):
        """Tests if an ``Id`` is a descendant of a catalog.

        arg:    id (osid.id.Id): an ``Id``
        arg:    catalog_id (osid.id.Id): the ``Id`` of a catalog
        return: (boolean) - ``true`` if the ``id`` is a descendant of
                the ``catalog_id,``  ``false`` otherwise
        raise:  NotFound - ``catalog_id`` not found
        raise:  NullArgument - ``catalog_id`` or ``id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*
        *implementation notes*: If ``id`` is not found return ``false``.

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.is_descendant_of_bin
        if self._catalog_session is not None:
            return self._catalog_session.is_descendant_of_catalog(id_=id_, catalog_id=catalog_id)
        return self._hierarchy_session.is_descendant(id_=id_, descendant_id=catalog_id)