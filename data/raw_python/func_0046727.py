def update_grade_entry(self, grade_entry_form):
        """Updates an existing grade entry.

        arg:    grade_entry_form (osid.grading.GradeEntryForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``grade_entry_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``grade_entry_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``grade_entry_form`` did not originate
                from ``get_grade_entry_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.update_resource_template
        collection = JSONClientValidated('grading',
                                         collection='GradeEntry',
                                         runtime=self._runtime)
        if not isinstance(grade_entry_form, ABCGradeEntryForm):
            raise errors.InvalidArgument('argument type is not an GradeEntryForm')
        if not grade_entry_form.is_for_update():
            raise errors.InvalidArgument('the GradeEntryForm is for update only, not create')
        try:
            if self._forms[grade_entry_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('grade_entry_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('grade_entry_form did not originate from this session')
        if not grade_entry_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(grade_entry_form._my_map)

        self._forms[grade_entry_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.GradeEntry(
            osid_object_map=grade_entry_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)