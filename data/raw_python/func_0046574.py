def is_descendant_of_family(self, id_, family_id):
        """Tests if an ``Id`` is a descendant of a family.

        arg:    id (osid.id.Id): an ``Id``
        arg:    family_id (osid.id.Id): the ``Id`` of a family
        return: (boolean) - ``true`` if the ``id`` is a descendant of
                the ``family_id,``  ``false`` otherwise
        raise:  NotFound - ``family_id`` is not found
        raise:  NullArgument - ``id`` or ``family_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*
        *implementation notes*: If ``id`` is not found return ``false``.

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.is_descendant_of_bin
        if self._catalog_session is not None:
            return self._catalog_session.is_descendant_of_catalog(id_=id_, catalog_id=family_id)
        return self._hierarchy_session.is_descendant(id_=id_, descendant_id=family_id)