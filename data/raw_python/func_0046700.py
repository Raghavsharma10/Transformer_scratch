def update_grade_system(self, grade_system_form):
        """Updates an existing grade system.

        arg:    grade_system_form (osid.grading.GradeSystemForm): the
                form containing the elements to be updated
        raise:  IllegalState - ``grade_system_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``grade_system_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``grade_system_form`` did not originate
                from ``get_grade_system_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.update_resource_template
        collection = JSONClientValidated('grading',
                                         collection='GradeSystem',
                                         runtime=self._runtime)
        if not isinstance(grade_system_form, ABCGradeSystemForm):
            raise errors.InvalidArgument('argument type is not an GradeSystemForm')
        if not grade_system_form.is_for_update():
            raise errors.InvalidArgument('the GradeSystemForm is for update only, not create')
        try:
            if self._forms[grade_system_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('grade_system_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('grade_system_form did not originate from this session')
        if not grade_system_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(grade_system_form._my_map)

        self._forms[grade_system_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.GradeSystem(
            osid_object_map=grade_system_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)