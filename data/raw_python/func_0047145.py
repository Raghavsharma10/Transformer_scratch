def create_objective(self, objective_form):
        """Creates a new ``Objective``.

        arg:    objective_form (osid.learning.ObjectiveForm): the form
                for this ``Objective``
        return: (osid.learning.Objective) - the new ``Objective``
        raise:  IllegalState - ``objective_form`` already used in a
                create transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``objective_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``objective_form`` did not originate from
                ``get_objective_form_for_create()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.create_resource_template
        collection = JSONClientValidated('learning',
                                         collection='Objective',
                                         runtime=self._runtime)
        if not isinstance(objective_form, ABCObjectiveForm):
            raise errors.InvalidArgument('argument type is not an ObjectiveForm')
        if objective_form.is_for_update():
            raise errors.InvalidArgument('the ObjectiveForm is for update only, not create')
        try:
            if self._forms[objective_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('objective_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('objective_form did not originate from this session')
        if not objective_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        insert_result = collection.insert_one(objective_form._my_map)

        self._forms[objective_form.get_id().get_identifier()] = CREATED
        result = objects.Objective(
            osid_object_map=collection.find_one({'_id': insert_result.inserted_id}),
            runtime=self._runtime,
            proxy=self._proxy)

        return result