def get_assessments_taken_by_search(self, assessment_taken_query, assessment_taken_search):
        """Pass through to provider AssessmentTakenSearchSession.get_assessments_taken_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_assessments_taken_by_search(assessment_taken_query, assessment_taken_search)