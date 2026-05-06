def get_assessment_part_form_for_update(self, assessment_part_id):
        """Gets the assessment part form for updating an existing assessment part.

        A new assessment part form should be requested for each update
        transaction.

        arg:    assessment_part_id (osid.id.Id): the ``Id`` of the
                ``AssessmentPart``
        return: (osid.assessment.authoring.AssessmentPartForm) - the
                assessment part form
        raise:  NotFound - ``assessment_part_id`` is not found
        raise:  NullArgument - ``assessment_part_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self._runtime)
        if not isinstance(assessment_part_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        if (assessment_part_id.get_identifier_namespace() != 'assessment_authoring.AssessmentPart' or
                assessment_part_id.get_authority() != self._authority):
            raise errors.InvalidArgument()
        result = collection.find_one({'_id': ObjectId(assessment_part_id.get_identifier())})

        mdata = {}
        if not result['assessmentPartId']:
            pass
        else:
            parent_part_id = Id(result['assessmentPartId'])
            mgr = self._get_provider_manager('ASSESSMENT_AUTHORING', local=True)
            lookup_session = mgr.get_assessment_part_lookup_session_for_bank(self._catalog_id, proxy=self._proxy)
            if lookup_session.get_assessment_parts_for_assessment_part(parent_part_id).available() > 1:
                mdata['sequestered']['is_read_only'] = True
                mdata['sequestered']['is_required'] = True
        obj_form = objects.AssessmentPartForm(osid_object_map=result,
                                              runtime=self._runtime,
                                              proxy=self._proxy,
                                              mdata=mdata)
        self._forms[obj_form.get_id().get_identifier()] = not UPDATED

        return obj_form