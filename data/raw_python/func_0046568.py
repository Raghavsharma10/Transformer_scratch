def get_parent_families(self, family_id):
        """Gets the parent families of the given ``id``.

        arg:    family_id (osid.id.Id): the ``Id`` of the ``Family`` to
                query
        return: (osid.relationship.FamilyList) - the parent families of
                the ``id``
        raise:  NotFound - a ``Family`` identified by ``Id is`` not
                found
        raise:  NullArgument - ``family_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_parent_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_parent_catalogs(catalog_id=family_id)
        return FamilyLookupSession(
            self._proxy,
            self._runtime).get_families_by_ids(
                list(self.get_parent_family_ids(family_id)))