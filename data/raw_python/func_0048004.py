def get_responses(self, assessment_taken_id):
        """Gets the submitted responses.

        arg:    assessment_taken_id (osid.id.Id): ``Id`` of the
                ``AssessmentTaken``
        return: (osid.assessment.ResponseList) - the submitted answers
        raise:  NotFound - ``assessment_taken_id`` is not found
        raise:  NullArgument - ``assessment_taken_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        taken_lookup_session = mgr.get_assessment_taken_lookup_session(proxy=self._proxy)
        taken_lookup_session.use_federated_bank_view()
        taken = taken_lookup_session.get_assessment_taken(assessment_taken_id)
        response_list = OsidListList()
        if 'sections' in taken._my_map:
            for section_id in taken._my_map['sections']:
                section = get_assessment_section(Id(section_id),
                                                 runtime=self._runtime,
                                                 proxy=self._proxy)
                response_list.append(section.get_responses())
        return ResponseList(response_list)