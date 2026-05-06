def update_hierarchy(self, hierarchy_form):
        """Updates an existing hierarchy.

        arg:    hierarchy_form (osid.hierarchy.HierarchyForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``hierarchy_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``hierarchy_id`` or ``hierarchy_form`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``hierarchy_form`` did not originate from
                ``get_hierarchy_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinAdminSession.update_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.update_catalog(catalog_form=hierarchy_form)
        collection = JSONClientValidated('hierarchy',
                                         collection='Hierarchy',
                                         runtime=self._runtime)
        if not isinstance(hierarchy_form, ABCHierarchyForm):
            raise errors.InvalidArgument('argument type is not an HierarchyForm')
        if not hierarchy_form.is_for_update():
            raise errors.InvalidArgument('the HierarchyForm is for update only, not create')
        try:
            if self._forms[hierarchy_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('hierarchy_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('hierarchy_form did not originate from this session')
        if not hierarchy_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(hierarchy_form._my_map)  # save is deprecated - change to replace_one

        self._forms[hierarchy_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned
        return objects.Hierarchy(osid_object_map=hierarchy_form._my_map, runtime=self._runtime, proxy=self._proxy)