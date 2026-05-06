def get_assessment_form_for_create(self, assessment_record_types):
        """Gets the assessment form for creating new assessments.

        A new form should be requested for each create transaction.

        arg:    assessment_record_types (osid.type.Type[]): array of
                assessment record types to be included in the create
                operation or an empty list if none
        return: (osid.assessment.AssessmentForm) - the assessment form
        raise:  NullArgument - ``assessment_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - unable to get form for requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.get_resource_form_for_create_template
        for arg in assessment_record_types:
            if not isinstance(arg, ABCType):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID Type')
        if assessment_record_types == []:
            obj_form = objects.AssessmentForm(
                bank_id=self._catalog_id,
                runtime=self._runtime,
                effective_agent_id=self.get_effective_agent_id(),
                proxy=self._proxy)
        else:
            obj_form = objects.AssessmentForm(
                bank_id=self._catalog_id,
                record_types=assessment_record_types,
                runtime=self._runtime,
                effective_agent_id=self.get_effective_agent_id(),
                proxy=self._proxy)
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form