def update_objective(self, objective_form):
        """Updates an existing objective.

        arg:    objective_form (osid.learning.ObjectiveForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``objective_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``objective_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``objective_form`` did not originate from
                ``get_objective_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.update_resource_template
        collection = JSONClientValidated('learning',
                                         collection='Objective',
                                         runtime=self._runtime)
        if not isinstance(objective_form, ABCObjectiveForm):
            raise errors.InvalidArgument('argument type is not an ObjectiveForm')
        if not objective_form.is_for_update():
            raise errors.InvalidArgument('the ObjectiveForm is for update only, not create')
        try:
            if self._forms[objective_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('objective_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('objective_form did not originate from this session')
        if not objective_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(objective_form._my_map)

        self._forms[objective_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.Objective(
            osid_object_map=objective_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)