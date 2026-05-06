def create_assessment(self, assessment_form):
        """Creates a new ``Assessment``.

        arg:    assessment_form (osid.assessment.AssessmentForm): the
                form for this ``Assessment``
        return: (osid.assessment.Assessment) - the new ``Assessment``
        raise:  IllegalState - ``assessment_form`` already used in a
                create transaction
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
                                         collection='Assessment',
                                         runtime=self._runtime)
        if not isinstance(assessment_form, ABCAssessmentForm):
            raise errors.InvalidArgument('argument type is not an AssessmentForm')
        if assessment_form.is_for_update():
            raise errors.InvalidArgument('the AssessmentForm is for update only, not create')
        try:
            if self._forms[assessment_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('assessment_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('assessment_form did not originate from this session')
        if not assessment_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        insert_result = collection.insert_one(assessment_form._my_map)

        self._forms[assessment_form.get_id().get_identifier()] = CREATED
        result = objects.Assessment(
            osid_object_map=collection.find_one({'_id': insert_result.inserted_id}),
            runtime=self._runtime,
            proxy=self._proxy)

        return result