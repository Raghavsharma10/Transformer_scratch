def has_parent_families(self, family_id):
        """Tests if the ``Family`` has any parents.

        arg:    family_id (osid.id.Id): the ``Id`` of a family
        return: (boolean) - ``true`` if the family has parents,
                ``false`` otherwise
        raise:  NotFound - ``family_id`` is not found
        raise:  NullArgument - ``family_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.has_parent_bins
        if self._catalog_session is not None:
            return self._catalog_session.has_parent_catalogs(catalog_id=family_id)
        return self._hierarchy_session.has_parents(id_=family_id)