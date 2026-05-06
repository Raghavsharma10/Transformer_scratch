def create_assessment_offered(self, assessment_offered_form):
        """Creates a new ``AssessmentOffered``.

        arg:    assessment_offered_form
                (osid.assessment.AssessmentOfferedForm): the form for
                this ``AssessmentOffered``
        return: (osid.assessment.AssessmentOffered) - the new
                ``AssessmentOffered``
        raise:  IllegalState - ``assessment_offrered_form`` already used
                in a create transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``assessment_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``assessment_form`` did not originate from
                ``get_assessment_form_for_create()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.create_resource_template
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentOffered',
                                         runtime=self._runtime)
        if not isinstance(assessment_offered_form, ABCAssessmentOfferedForm):
            raise errors.InvalidArgument('argument type is not an AssessmentOfferedForm')
        if assessment_offered_form.is_for_update():
            raise errors.InvalidArgument('the AssessmentOfferedForm is for update only, not create')
        try:
            if self._forms[assessment_offered_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('assessment_offered_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('assessment_offered_form did not originate from this session')
        if not assessment_offered_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        insert_result = collection.insert_one(assessment_offered_form._my_map)

        self._forms[assessment_offered_form.get_id().get_identifier()] = CREATED
        result = objects.AssessmentOffered(
            osid_object_map=collection.find_one({'_id': insert_result.inserted_id}),
            runtime=self._runtime,
            proxy=self._proxy)

        return result