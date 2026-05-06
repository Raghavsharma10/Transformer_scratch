def get_assessments(self):
        """Gets any assessments associated with this activity.

        return: (osid.assessment.AssessmentList) - list of assessments
        raise:  IllegalState - ``is_assessment_based_activity()`` is
                ``false``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_assets_template
        if not bool(self._my_map['assessmentIds']):
            raise errors.IllegalState('no assessmentIds')
        mgr = self._get_provider_manager('ASSESSMENT')
        if not mgr.supports_assessment_lookup():
            raise errors.OperationFailed('Assessment does not support Assessment lookup')

        # What about the Proxy?
        lookup_session = mgr.get_assessment_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_bank_view()
        return lookup_session.get_assessments_by_ids(self.get_assessment_ids())