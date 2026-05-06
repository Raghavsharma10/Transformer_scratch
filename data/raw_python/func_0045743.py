def get_families_by_genus_type(self, family_genus_type=None):
        """Gets a ``FamilyList`` corresponding to the given family genus ``Type`` which
            does not include families of genus types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known families
        or an error results. Otherwise, the returned list may contain
        only those families that are accessible through this session.

        arg:    family_genus_type (osid.type.Type): a family genus type
        return: (osid.relationship.FamilyList) - the returned ``Family
                list``
        raise:  NullArgument - ``family_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if family_genus_type is None:
            raise NullArgument()
        url_path = '/handcar/services/relationship/families'
        families_of_type = []
        all_families = self._get_request(url_path)
        for family in all_families:
            # DO WE NEED TO CHECK ALL THREE ATRIBUTES OF THE Id HERE?
            if family['genusTypeId'] == str(family_genus_type):
                families_of_type.append(family)
        return objects.FamilyList(families_of_type)