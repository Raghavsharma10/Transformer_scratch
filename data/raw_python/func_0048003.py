def get_items(self, assessment_taken_id):
        """Gets the items questioned in a assessment.

        arg:    assessment_taken_id (osid.id.Id): ``Id`` of the
                ``AssessmentTaken``
        return: (osid.assessment.ItemList) - the list of assessment
                questions
        raise:  NotFound - ``assessment_taken_id`` is not found
        raise:  NullArgument - ``assessment_taken_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        taken_lookup_session = mgr.get_assessment_taken_lookup_session(proxy=self._proxy)
        taken_lookup_session.use_federated_bank_view()
        taken = taken_lookup_session.get_assessment_taken(assessment_taken_id)
        ils = get_item_lookup_session(runtime=self._runtime, proxy=self._proxy)
        ils.use_federated_bank_view()
        item_list = []
        if 'sections' in taken._my_map:
            for section_id in taken._my_map['sections']:
                section = get_assessment_section(Id(section_id),
                                                 runtime=self._runtime,
                                                 proxy=self._proxy)
                for question in section._my_map['questions']:
                    item_list.append(ils.get_item(Id(question['questionId'])))
        return ItemList(item_list)