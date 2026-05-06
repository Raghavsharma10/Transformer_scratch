def get_families_by_ids(self, family_ids=None):
        """Gets a ``FamilyList`` corresponding to the given ``IdList``.

        In plenary mode, the returned list contains all of the families
        specified in the ``Id`` list, in the order of the list,
        including duplicates, or an error results if an ``Id`` in the
        supplied list is not found or inaccessible. Otherwise,
        inaccessible families may be omitted from the list and may
        present the elements in any order including returning a unique
        set.

        arg:    family_ids (osid.id.IdList): the list of ``Ids`` to
                retrieve
        return: (osid.relationship.FamilyList) - the returned ``Family
                list``
        raise:  NotFound - an ``Id was`` not found
        raise:  NullArgument - ``family_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if family_ids is None:
            raise NullArgument()
        families = []
        for i in family_ids:
            family = None
            url_path = '/handcar/services/relationship/families/' + str(i)
            try:
                family = self._get_request(url_path)
            except (NotFound, OperationFailed):
                if self._family_view == PLENARY:
                    raise
                else:
                    pass
            if family:
                if not (self._family_view == COMPARATIVE and
                        family in families):
                    families.append(family)
        return objects.FamilykList(families)