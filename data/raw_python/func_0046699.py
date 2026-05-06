def create_grade_system(self, grade_system_form):
        """Creates a new ``GradeSystem``.

        arg:    grade_system_form (osid.grading.GradeSystemForm): the
                form for this ``GradeSystem``
        return: (osid.grading.GradeSystem) - the new ``GradeSystem``
        raise:  IllegalState - ``grade_system_form`` already used in a
                create transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``grade_system_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``grade_system_form`` did not originate
                from ``get_grade_system_form_for_create()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.create_resource_template
        collection = JSONClientValidated('grading',
                                         collection='GradeSystem',
                                         runtime=self._runtime)
        if not isinstance(grade_system_form, ABCGradeSystemForm):
            raise errors.InvalidArgument('argument type is not an GradeSystemForm')
        if grade_system_form.is_for_update():
            raise errors.InvalidArgument('the GradeSystemForm is for update only, not create')
        try:
            if self._forms[grade_system_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('grade_system_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('grade_system_form did not originate from this session')
        if not grade_system_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        insert_result = collection.insert_one(grade_system_form._my_map)

        self._forms[grade_system_form.get_id().get_identifier()] = CREATED
        result = objects.GradeSystem(
            osid_object_map=collection.find_one({'_id': insert_result.inserted_id}),
            runtime=self._runtime,
            proxy=self._proxy)

        return result