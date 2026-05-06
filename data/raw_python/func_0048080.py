def get_assessment_taken_form_for_create(self, assessment_offered_id, assessment_taken_record_types):
        """Gets the assessment taken form for creating new assessments taken.

        A new form should be requested for each create transaction.

        arg:    assessment_offered_id (osid.id.Id): the ``Id`` of the
                related ``AssessmentOffered``
        arg:    assessment_taken_record_types (osid.type.Type[]): array
                of assessment taken record types to be included in the
                create operation or an empty list if none
        return: (osid.assessment.AssessmentTakenForm) - the assessment
                taken form
        raise:  NotFound - ``assessment_offered_id`` is not found
        raise:  NullArgument - ``assessment_offered_id`` or
                ``assessment_taken_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - unable to get form for requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        if not isinstance(assessment_offered_id, ABCId):
            raise errors.InvalidArgument('argument is not a valid OSID Id')
        for arg in assessment_taken_record_types:
            if not isinstance(arg, ABCType):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID Type')

        am = self._get_provider_manager('ASSESSMENT')
        aols = am.get_assessment_offered_lookup_session(proxy=self._proxy)
        aols.use_federated_bank_view()
        offered = aols.get_assessment_offered(assessment_offered_id)
        try:
            deadline = offered.get_deadline()
            nowutc = DateTime.utcnow()
            if nowutc > deadline:
                raise errors.PermissionDenied('you are passed the deadline for the offered')
        except errors.IllegalState:
            # no deadline set
            pass

        if assessment_taken_record_types == []:
            # WHY are we passing bank_id = self._catalog_id below, seems redundant:
            obj_form = objects.AssessmentTakenForm(
                bank_id=self._catalog_id,
                assessment_offered_id=assessment_offered_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        else:
            obj_form = objects.AssessmentTakenForm(
                bank_id=self._catalog_id,
                record_types=assessment_taken_record_types,
                assessment_offered_id=assessment_offered_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        obj_form._for_update = False
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form