def update_objective_bank(self, objective_bank_form=None):
        """Updates an existing objective bank.

        arg:    objectiveBankForm (osid.learning.ObjectiveBankForm): the
                form containing the elements to be updated
        raise:  IllegalState - objectiveBankForm already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - objectiveBankForm is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - objective_bank_form did not originate from
                get_objective_bank_form_for_update()
        compliance: mandatory - This method must be implemented.

        """
        if objective_bank_form is None:
            raise NullArgument()
        if not isinstance(objective_bank_form, abc_learning_objects.ObjectiveBankForm):
            raise InvalidArgument('argument type is not an ObjectiveBankForm')
        if not objective_bank_form.is_for_update():
            raise InvalidArgument('form is for create only, not update')

        # Check for "sandbox" genus type.  Hardcoded for now:
        # if objective_bank_form._my_map['genusTypeId'] != 'mc3-objectivebank%3Amc3.learning.objectivebank.sandbox%40MIT-OEIT':
        #    raise PermissionDenied('Handcar only supports updating \'sandbox\' type ObjectiveBanks')

        try:
            if self._forms[objective_bank_form.get_id().get_identifier()] == UPDATED:
                raise IllegalState('form already used in an update transaction')
        except KeyError:
            raise Unsupported('form did not originate from this session')
        if not objective_bank_form.is_valid():
            raise InvalidArgument('one or more of the form elements is invalid')

        url_path = construct_url('objective_banks')
        try:
            result = self._put_request(url_path, objective_bank_form._my_map)
        except Exception:
            raise  # OperationFailed
        self._forms[objective_bank_form.get_id().get_identifier()] = UPDATED
        return objects.ObjectiveBank(result)