def create_grade_entry(self, grade_entry_form):
        """Creates a new ``GradeEntry``.

        arg:    grade_entry_form (osid.grading.GradeEntryForm): the form
                for this ``GradeEntry``
        return: (osid.grading.GradeEntry) - the new ``GradeEntry``
        raise:  IllegalState - ``grade_entry_form`` already used in a
                create transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``grade_entry_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``grade_entry_form`` did not originate
                from ``get_grade_entry_form_for_create()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.create_resource_template
        collection = JSONClientValidated('grading',
                                         collection='GradeEntry',
                                         runtime=self._runtime)
        if not isinstance(grade_entry_form, ABCGradeEntryForm):
            raise errors.InvalidArgument('argument type is not an GradeEntryForm')
        if grade_entry_form.is_for_update():
            raise errors.InvalidArgument('the GradeEntryForm is for update only, not create')
        try:
            if self._forms[grade_entry_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('grade_entry_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('grade_entry_form did not originate from this session')
        if not grade_entry_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        insert_result = collection.insert_one(grade_entry_form._my_map)

        self._forms[grade_entry_form.get_id().get_identifier()] = CREATED
        result = objects.GradeEntry(
            osid_object_map=collection.find_one({'_id': insert_result.inserted_id}),
            runtime=self._runtime,
            proxy=self._proxy)

        return result