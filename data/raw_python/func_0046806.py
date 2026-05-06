def create_objective(self, objective_form=None):
        """Creates a new Objective.

        arg:    objectiveForm (osid.learning.ObjectiveForm): the form
                for this Objective
        return: (osid.learning.Objective) - the new Objective
        raise:  IllegalState - objectiveForm already used in a create
                transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - objectiveForm is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - objectiveForm did not originate from
                get_objective_form_for_create()
        compliance: mandatory - This method must be implemented.

        """
        if objective_form is None:
            raise NullArgument()
        if not isinstance(objective_form, abc_learning_objects.ObjectiveForm):
            raise InvalidArgument('argument type is not an ObjectiveForm')
        if objective_form.is_for_update():
            raise InvalidArgument('form is for update only, not create')
        try:
            if self._forms[objective_form.get_id().get_identifier()] == CREATED:
                raise IllegalState('form already used in a create transaction')
        except KeyError:
            raise Unsupported('form did not originate from this session')
        if not objective_form.is_valid():
            raise InvalidArgument('one or more of the form elements is invalid')

        url_path = construct_url('objectives',
                                 bank_id=self._catalog_idstr)
        try:
            result = self._post_request(url_path, objective_form._my_map)
        except Exception:
            raise  # OperationFailed
        self._forms[objective_form.get_id().get_identifier()] = CREATED
        return objects.Objective(result)