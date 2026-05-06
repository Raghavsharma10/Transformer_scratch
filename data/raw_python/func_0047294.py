def get_assessment_part(self):
        """Gets the assessment part to which this rule belongs.

        return: (osid.assessment.authoring.AssessmentPart) - an
                assessment part
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective
        if not bool(self._my_map['assessmentPartId']):
            raise errors.IllegalState('assessment_part empty')
        mgr = self._get_provider_manager('ASSESSMENT_AUTHORING')
        if not mgr.supports_assessment_part_lookup():
            raise errors.OperationFailed('Assessment_Authoring does not support AssessmentPart lookup')
        lookup_session = mgr.get_assessment_part_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_bank_view()
        return lookup_session.get_assessment_part(self.get_assessment_part_id())