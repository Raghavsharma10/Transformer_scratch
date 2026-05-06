def create_assessment_part_for_assessment(self, assessment_part_form):
        """Creates a new assessment part.

        arg:    assessment_part_form
                (osid.assessment.authoring.AssessmentPartForm):
                assessment part form
        return: (osid.assessment.authoring.AssessmentPart) - the new
                part
        raise:  IllegalState - ``assessment_part_form`` already used in
                a create transaction
        raise:  InvalidArgument - ``assessment_part_form`` is invalid
        raise:  NullArgument - ``assessment_part_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``assessment_part_form`` did not originate
                from
                ``get_assessment_part_form_for_create_for_assessment()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.create_resource_template
        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self._runtime)
        if not isinstance(assessment_part_form, ABCAssessmentPartForm):
            raise errors.InvalidArgument('argument type is not an AssessmentPartForm')
        if assessment_part_form.is_for_update():
            raise errors.InvalidArgument('the AssessmentPartForm is for update only, not create')
        try:
            if self._forms[assessment_part_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('assessment_part_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('assessment_part_form did not originate from this session')
        if not assessment_part_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        insert_result = collection.insert_one(assessment_part_form._my_map)

        self._forms[assessment_part_form.get_id().get_identifier()] = CREATED
        result = objects.AssessmentPart(
            osid_object_map=collection.find_one({'_id': insert_result.inserted_id}),
            runtime=self._runtime,
            proxy=self._proxy)

        return result