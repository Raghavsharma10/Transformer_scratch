def update_activity(self, activity_form):
        """Updates an existing activity,.

        arg:    activity_form (osid.learning.ActivityForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``activity_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``activity_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``activity_form`` did not originate from
                ``get_activity_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.update_resource_template
        collection = JSONClientValidated('learning',
                                         collection='Activity',
                                         runtime=self._runtime)
        if not isinstance(activity_form, ABCActivityForm):
            raise errors.InvalidArgument('argument type is not an ActivityForm')
        if not activity_form.is_for_update():
            raise errors.InvalidArgument('the ActivityForm is for update only, not create')
        try:
            if self._forms[activity_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('activity_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('activity_form did not originate from this session')
        if not activity_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(activity_form._my_map)

        self._forms[activity_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.Activity(
            osid_object_map=activity_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)