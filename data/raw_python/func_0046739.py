def update_gradebook_column(self, gradebook_column_form):
        """Updates an existing gradebook column.

        arg:    gradebook_column_form
                (osid.grading.GradebookColumnForm): the form containing
                the elements to be updated
        raise:  IllegalState - ``gradebook_column_form`` already used in
                an update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``gradebook_column_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``gradebook_column_form`` did not
                originate from
                ``get_gradebook_column_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        collection = JSONClientValidated('grading',
                                         collection='GradebookColumn',
                                         runtime=self._runtime)
        if not isinstance(gradebook_column_form, ABCGradebookColumnForm):
            raise errors.InvalidArgument('argument type is not an GradebookColumnForm')
        if not gradebook_column_form.is_for_update():
            raise errors.InvalidArgument('the GradebookColumnForm is for update only, not create')
        try:
            if self._forms[gradebook_column_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('gradebook_column_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('gradebook_column_form did not originate from this session')
        if not gradebook_column_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')

        # check that there are no entries, if updating the gradeSystemId
        old_column = collection.find_one({"_id": gradebook_column_form._my_map['_id']})
        if old_column['gradeSystemId'] != gradebook_column_form._my_map['gradeSystemId']:
            if self._has_entries(gradebook_column_form.id_):
                raise errors.IllegalState('Entries exist in this gradebook column. ' +
                                          'Cannot change the grade system.')

        collection.save(gradebook_column_form._my_map)

        self._forms[gradebook_column_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.GradebookColumn(
            osid_object_map=gradebook_column_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)