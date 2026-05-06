def get_assessment_part_form_for_create_for_assessment(self, assessment_id, assessment_part_record_types):
        """Gets the assessment part form for creating new assessment parts for an assessment.

        A new form should be requested for each create transaction.

        arg:    assessment_id (osid.id.Id): an assessment ``Id``
        arg:    assessment_part_record_types (osid.type.Type[]): array
                of assessment part record types to be included in the
                create operation or an empty list if none
        return: (osid.assessment.authoring.AssessmentPartForm) - the
                assessment part form
        raise:  NotFound - ``assessment_id`` is not found
        raise:  NullArgument - ``assessment_id`` or
                ``assessment_part_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - unable to get form for requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.learning.ActivityAdminSession.get_activity_form_for_create_template

        if not isinstance(assessment_id, ABCId):
            raise errors.InvalidArgument('argument is not a valid OSID Id')
        for arg in assessment_part_record_types:
            if not isinstance(arg, ABCType):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID Type')
        if assessment_part_record_types == []:
            # WHY are we passing bank_id = self._catalog_id below, seems redundant:
            obj_form = objects.AssessmentPartForm(
                bank_id=self._catalog_id,
                assessment_id=assessment_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        else:
            obj_form = objects.AssessmentPartForm(
                bank_id=self._catalog_id,
                record_types=assessment_part_record_types,
                assessment_id=assessment_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        obj_form._for_update = False
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form