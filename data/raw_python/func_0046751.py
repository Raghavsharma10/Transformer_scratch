def update_gradebook(self, gradebook_form):
        """Updates an existing gradebook.

        arg:    gradebook_form (osid.grading.GradebookForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``gradebook_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``gradebook_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``gradebook_form did not originate from
                get_gradebook_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinAdminSession.update_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.update_catalog(catalog_form=gradebook_form)
        collection = JSONClientValidated('grading',
                                         collection='Gradebook',
                                         runtime=self._runtime)
        if not isinstance(gradebook_form, ABCGradebookForm):
            raise errors.InvalidArgument('argument type is not an GradebookForm')
        if not gradebook_form.is_for_update():
            raise errors.InvalidArgument('the GradebookForm is for update only, not create')
        try:
            if self._forms[gradebook_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('gradebook_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('gradebook_form did not originate from this session')
        if not gradebook_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(gradebook_form._my_map)  # save is deprecated - change to replace_one

        self._forms[gradebook_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned
        return objects.Gradebook(osid_object_map=gradebook_form._my_map, runtime=self._runtime, proxy=self._proxy)