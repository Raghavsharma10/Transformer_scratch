def get_sequence_rules_for_assessment(self, assessment_id):
        """Gets a ``SequenceRuleList`` for an entire assessment.

        arg:    assessment_id (osid.id.Id): an assessment ``Id``
        return: (osid.assessment.authoring.SequenceRuleList) - the
                returned ``SequenceRule`` list
        raise:  NullArgument - ``assessment_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # First, recursively get all the partIds for the assessment
        def get_all_children_part_ids(part):
            child_ids = []
            if part.has_children():
                child_ids = list(part.get_child_assessment_part_ids())
                for child in part.get_child_assessment_parts():
                    child_ids += get_all_children_part_ids(child)
            return child_ids

        all_assessment_part_ids = []

        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        lookup_session = mgr.get_assessment_lookup_session(proxy=self._proxy)
        lookup_session.use_federated_bank_view()
        assessment = lookup_session.get_assessment(assessment_id)

        if assessment.has_children():
            mgr = self._get_provider_manager('ASSESSMENT_AUTHORING', local=True)
            lookup_session = mgr.get_assessment_part_lookup_session(proxy=self._proxy)
            lookup_session.use_federated_bank_view()
            all_assessment_part_ids = list(assessment.get_child_ids())
            for child_part_id in assessment.get_child_ids():
                child_part = lookup_session.get_assessment_part(child_part_id)
                all_assessment_part_ids += get_all_children_part_ids(child_part)

        id_strs = [str(part_id) for part_id in all_assessment_part_ids]
        collection = JSONClientValidated('assessment_authoring',
                                         collection='SequenceRule',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'assessmentPartId': {'$in': id_strs}},
                 **self._view_filter()))
        return objects.SequenceRuleList(result, runtime=self._runtime)