def update_assessment(self, assessment_form):
        """Updates an existing assessment.

        arg:    assessment_form (osid.assessment.AssessmentForm): the
                form containing the elements to be updated
        raise:  IllegalState - ``assessment_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``assessment_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``assessment_form did not originate from
                get_assessment_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.update_resource_template
        collection = JSONClientValidated('assessment',
                                         collection='Assessment',
                                         runtime=self._runtime)
        if not isinstance(assessment_form, ABCAssessmentForm):
            raise errors.InvalidArgument('argument type is not an AssessmentForm')
        if not assessment_form.is_for_update():
            raise errors.InvalidArgument('the AssessmentForm is for update only, not create')
        try:
            if self._forms[assessment_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('assessment_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('assessment_form did not originate from this session')
        if not assessment_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(assessment_form._my_map)

        self._forms[assessment_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.Assessment(
            osid_object_map=assessment_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)