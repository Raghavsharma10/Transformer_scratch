def update_assessment_part(self, assessment_part_id, assessment_part_form):
        """Updates an existing assessment part.

        arg:    assessment_part_id (osid.id.Id): the ``Id`` of the
                ``AssessmentPart``
        arg:    assessment_part_form
                (osid.assessment.authoring.AssessmentPartForm): part
                form
        raise:  NotFound - ``assessment_part_id`` not found
        raise:  NullArgument - ``assessment_part_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``assessment_part_form`` is not of this
                service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.update_resource_template
        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self._runtime)
        if not isinstance(assessment_part_form, ABCAssessmentPartForm):
            raise errors.InvalidArgument('argument type is not an AssessmentPartForm')
        if not assessment_part_form.is_for_update():
            raise errors.InvalidArgument('the AssessmentPartForm is for update only, not create')
        try:
            if self._forms[assessment_part_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('assessment_part_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('assessment_part_form did not originate from this session')
        if not assessment_part_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(assessment_part_form._my_map)

        self._forms[assessment_part_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.AssessmentPart(
            osid_object_map=assessment_part_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)