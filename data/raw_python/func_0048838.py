def get_assessment_part_form_for_create_for_assessment_part(self, assessment_part_id, assessment_part_record_types):
        """Gets the assessment part form for creating new assessment parts under another assessment part.

        A new form should be requested for each create transaction.

        arg:    assessment_part_id (osid.id.Id): an assessment part
                ``Id``
        arg:    assessment_part_record_types (osid.type.Type[]): array
                of assessment part record types to be included in the
                create operation or an empty list if none
        return: (osid.assessment.authoring.AssessmentPartForm) - the
                assessment part form
        raise:  NotFound - ``assessment_part_id`` is not found
        raise:  NullArgument - ``assessment_part_id`` or
                ``assessment_part_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - unable to get form for requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        if not isinstance(assessment_part_id, ABCId):
            raise errors.InvalidArgument('argument is not a valid OSID Id')
        for arg in assessment_part_record_types:
            if not isinstance(arg, ABCType):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID Type')
        if assessment_part_record_types == []:
            assessment_part_record_types = None
        mgr = self._get_provider_manager('ASSESSMENT_AUTHORING', local=True)
        lookup_session = mgr.get_assessment_part_lookup_session_for_bank(self._catalog_id, proxy=self._proxy)
        child_parts = lookup_session.get_assessment_parts_for_assessment_part(assessment_part_id)
        mdata = {}
        # Check for underlying Parts, whether Sections and set appropriate mdata overrides:
        if child_parts.available == 0:
            pass
        else:
            mdata['sequestered'] = {}
            mdata['sequestered']['is_read_only'] = True
            mdata['sequestered']['is_required'] = True
            if child_parts.available() > 0 and child_parts.next().is_section():
                mdata['sequestered']['default_boolean_values'] = [False]
            else:
                mdata['sequestered']['default_boolean_values'] = [True]
        # WHY are we passing bank_id = self._catalog_id below, seems redundant:
        obj_form = objects.AssessmentPartForm(
            bank_id=self._catalog_id,
            record_types=assessment_part_record_types,
            assessment_part_id=assessment_part_id,
            catalog_id=self._catalog_id,
            runtime=self._runtime,
            mdata=mdata)
        obj_form._for_update = False
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form