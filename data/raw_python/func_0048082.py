def update_assessment_taken(self, assessment_taken_form):
        """Updates an existing assessment taken.

        arg:    assessment_taken_form
                (osid.assessment.AssessmentTakenForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``assessment_taken_form`` already used in
                an update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``assessment_taken_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``assessment_offered_form`` did not
                originate from
                ``get_assessment_taken_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.update_resource_template
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentTaken',
                                         runtime=self._runtime)
        if not isinstance(assessment_taken_form, ABCAssessmentTakenForm):
            raise errors.InvalidArgument('argument type is not an AssessmentTakenForm')
        if not assessment_taken_form.is_for_update():
            raise errors.InvalidArgument('the AssessmentTakenForm is for update only, not create')
        try:
            if self._forms[assessment_taken_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('assessment_taken_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('assessment_taken_form did not originate from this session')
        if not assessment_taken_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(assessment_taken_form._my_map)

        self._forms[assessment_taken_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.AssessmentTaken(
            osid_object_map=assessment_taken_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)