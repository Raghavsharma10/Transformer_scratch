def create_activity(self, activity_form):
        """Creates a new ``Activity``.

        arg:    activity_form (osid.learning.ActivityForm): the form for
                this ``Activity``
        return: (osid.learning.Activity) - the new ``Activity``
        raise:  IllegalState - ``activity_form`` already used in a
                create transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``activity_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``activity_form`` did not originate from
                ``get_activity_form_for_create()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.create_resource_template
        collection = JSONClientValidated('learning',
                                         collection='Activity',
                                         runtime=self._runtime)
        if not isinstance(activity_form, ABCActivityForm):
            raise errors.InvalidArgument('argument type is not an ActivityForm')
        if activity_form.is_for_update():
            raise errors.InvalidArgument('the ActivityForm is for update only, not create')
        try:
            if self._forms[activity_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('activity_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('activity_form did not originate from this session')
        if not activity_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        insert_result = collection.insert_one(activity_form._my_map)

        self._forms[activity_form.get_id().get_identifier()] = CREATED
        result = objects.Activity(
            osid_object_map=collection.find_one({'_id': insert_result.inserted_id}),
            runtime=self._runtime,
            proxy=self._proxy)

        return result