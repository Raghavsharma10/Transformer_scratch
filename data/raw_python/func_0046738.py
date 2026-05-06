def create_gradebook_column(self, gradebook_column_form):
        """Creates a new ``GradebookColumn``.

        arg:    gradebook_column_form
                (osid.grading.GradebookColumnForm): the form for this
                ``GradebookColumn``
        return: (osid.grading.GradebookColumn) - the new
                ``GradebookColumn``
        raise:  IllegalState - ``gradebook_column_form`` already used in
                a create transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``gradebook_column_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``gradebook_column_form`` did not
                originate from
                ``get_gradebook_column_form_for_create()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.create_resource_template
        collection = JSONClientValidated('grading',
                                         collection='GradebookColumn',
                                         runtime=self._runtime)
        if not isinstance(gradebook_column_form, ABCGradebookColumnForm):
            raise errors.InvalidArgument('argument type is not an GradebookColumnForm')
        if gradebook_column_form.is_for_update():
            raise errors.InvalidArgument('the GradebookColumnForm is for update only, not create')
        try:
            if self._forms[gradebook_column_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('gradebook_column_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('gradebook_column_form did not originate from this session')
        if not gradebook_column_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        insert_result = collection.insert_one(gradebook_column_form._my_map)

        self._forms[gradebook_column_form.get_id().get_identifier()] = CREATED
        result = objects.GradebookColumn(
            osid_object_map=collection.find_one({'_id': insert_result.inserted_id}),
            runtime=self._runtime,
            proxy=self._proxy)

        return result