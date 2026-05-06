def create_sequence_rule(self, sequence_rule_form):
        """Creates a new ``SequenceRule``.

        arg:    sequence_rule_form
                (osid.assessment.authoring.SequenceRuleForm): the form
                for this ``SequenceRule``
        return: (osid.assessment.authoring.SequenceRule) - the new
                ``SequenceRule``
        raise:  IllegalState - ``sequence_rule_form`` already used in a
                create transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``sequence_rule_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``sequence_rule_form`` did not originate
                from ``get_sequence_rule_form_for_create()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.create_resource_template
        collection = JSONClientValidated('assessment_authoring',
                                         collection='SequenceRule',
                                         runtime=self._runtime)
        if not isinstance(sequence_rule_form, ABCSequenceRuleForm):
            raise errors.InvalidArgument('argument type is not an SequenceRuleForm')
        if sequence_rule_form.is_for_update():
            raise errors.InvalidArgument('the SequenceRuleForm is for update only, not create')
        try:
            if self._forms[sequence_rule_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('sequence_rule_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('sequence_rule_form did not originate from this session')
        if not sequence_rule_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        insert_result = collection.insert_one(sequence_rule_form._my_map)

        self._forms[sequence_rule_form.get_id().get_identifier()] = CREATED
        result = objects.SequenceRule(
            osid_object_map=collection.find_one({'_id': insert_result.inserted_id}),
            runtime=self._runtime,
            proxy=self._proxy)

        return result