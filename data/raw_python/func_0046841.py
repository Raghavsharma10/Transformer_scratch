def update_activity(self, activity_form=None):
        """Updates an existing activity,.

        arg:    activityForm (osid.learning.ActivityForm): the form
                containing the elements to be updated
        raise:  IllegalState - activityForm already used in an update
                transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - activityForm is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - activityForm did not originate from
                get_activity_form_for_update()
        compliance: mandatory - This method must be implemented.

        """
        if activity_form is None:
            raise NullArgument()
        if not isinstance(activity_form, abc_learning_objects.ActivityForm):
            raise InvalidArgument('argument type is not an ActivityForm')
        if not activity_form.is_for_update():
            raise InvalidArgument('form is for create only, not update')
        try:
            if self._forms[activity_form.get_id().get_identifier()] == UPDATED:
                raise IllegalState('form already used in an update transaction')
        except KeyError:
            raise Unsupported('form did not originate from this session')
        if not activity_form.is_valid():
            raise InvalidArgument('one or more of the form elements is invalid')

        url_path = construct_url('activities',
                                 bank_id=self._catalog_idstr)
        try:
            result = self._put_request(url_path, activity_form._my_map)
        except Exception:
            raise  # OperationFailed
        self._forms[activity_form.get_id().get_identifier()] = UPDATED
        return objects.Activity(result)