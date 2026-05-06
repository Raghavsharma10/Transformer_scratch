def get_grade_entry_form_for_update(self, grade_entry_id):
        """Gets the grade entry form for updating an existing entry.

        A new grade entry form should be requested for each update
        transaction.

        arg:    grade_entry_id (osid.id.Id): the ``Id`` of the
                ``GradeEntry``
        return: (osid.grading.GradeEntryForm) - the grade entry form
        raise:  NotFound - ``grade_entry_id`` is not found
        raise:  NullArgument - ``grade_entry_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        collection = JSONClientValidated('grading',
                                         collection='GradeEntry',
                                         runtime=self._runtime)
        if not isinstance(grade_entry_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        if (grade_entry_id.get_identifier_namespace() != 'grading.GradeEntry' or
                grade_entry_id.get_authority() != self._authority):
            raise errors.InvalidArgument()
        result = collection.find_one({'_id': ObjectId(grade_entry_id.get_identifier())})

        obj_form = objects.GradeEntryForm(
            osid_object_map=result,
            effective_agent_id=str(self.get_effective_agent_id()),
            runtime=self._runtime,
            proxy=self._proxy)
        self._forms[obj_form.get_id().get_identifier()] = not UPDATED

        return obj_form