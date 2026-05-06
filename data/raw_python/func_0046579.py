def remove_child_family(self, family_id, child_id):
        """Removes a child from a family.

        arg:    family_id (osid.id.Id): the ``Id`` of a family
        arg:    child_id (osid.id.Id): the ``Id`` of the new child
        raise:  NotFound - ``family_id`` not a parent of ``child_id``
        raise:  NullArgument - ``family_id`` or ``child_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.remove_child_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.remove_child_catalog(catalog_id=family_id, child_id=child_id)
        return self._hierarchy_session.remove_child(id_=family_id, child_id=child_id)