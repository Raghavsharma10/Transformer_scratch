def create_assessment_taken(self, assessment_taken_form):
        """Creates a new ``AssessmentTaken``.

        arg:    assessment_taken_form
                (osid.assessment.AssessmentTakenForm): the form for this
                ``AssessmentTaken``
        return: (osid.assessment.AssessmentTaken) - the new
                ``AssessmentTaken``
        raise:  IllegalState - ``assessment_taken_form`` already used in
                a create transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``assessment_taken_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``assessment_offered_form`` did not
                originate from
                ``get_assessment_taken_form_for_create()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # This impl differs from the usual create_osid_object method in that it
        # sets an agent id and default display name based on the underlying Assessment
        # and checks for exceeding max attempts...
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentTaken',
                                         runtime=self._runtime)
        if not isinstance(assessment_taken_form, ABCAssessmentTakenForm):
            raise errors.InvalidArgument('argument type is not an AssessmentTakenForm')
        if assessment_taken_form.is_for_update():
            raise errors.InvalidArgument('the AssessmentForm is for update only, not create')
        try:
            if self._forms[assessment_taken_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('assessment_taken_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('assessment_taken_form did not originate from this session')
        if not assessment_taken_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')

        # ...here:
        assessment_offered_id = Id(assessment_taken_form._my_map['assessmentOfferedId'])
        aols = AssessmentOfferedLookupSession(
            catalog_id=None, runtime=self._runtime)
        aols.use_federated_bank_view()
        assessment_offered = aols.get_assessment_offered(assessment_offered_id)
        try:
            if assessment_offered.has_max_attempts():
                max_attempts = assessment_offered.get_max_attempts()
                num_takens = collection.find({'$and': [{'assessmentOfferedId': str(assessment_offered.get_id())},
                                                       {'takingAgentId': str(self.get_effective_agent_id())},
                                                       {'assigned' + self._catalog_name + 'Ids': {'$in': [str(self._catalog_id)]}}]}).count()
                if num_takens >= max_attempts:
                    raise errors.PermissionDenied('exceeded max attempts')
        except AttributeError:
            pass
        assessment_taken_form._my_map['takingAgentId'] = str(self.get_effective_agent_id())

        insert_result = collection.insert_one(assessment_taken_form._my_map)
        self._forms[assessment_taken_form.get_id().get_identifier()] = CREATED
        return objects.AssessmentTaken(
            osid_object_map=collection.find_one({'_id': insert_result.inserted_id}),
            runtime=self._runtime,
            proxy=self._proxy)