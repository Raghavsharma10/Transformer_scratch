def update_family(self, family_form):
        """Updates an existing family.

        arg:    family_form (osid.relationship.FamilyForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``family_form`` already used in an update
                transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``family_id`` or ``family_form`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``family_form`` did not originate from
                ``get_family_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinAdminSession.update_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.update_catalog(catalog_form=family_form)
        collection = JSONClientValidated('relationship',
                                         collection='Family',
                                         runtime=self._runtime)
        if not isinstance(family_form, ABCFamilyForm):
            raise errors.InvalidArgument('argument type is not an FamilyForm')
        if not family_form.is_for_update():
            raise errors.InvalidArgument('the FamilyForm is for update only, not create')
        try:
            if self._forms[family_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('family_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('family_form did not originate from this session')
        if not family_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(family_form._my_map)  # save is deprecated - change to replace_one

        self._forms[family_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned
        return objects.Family(osid_object_map=family_form._my_map, runtime=self._runtime, proxy=self._proxy)