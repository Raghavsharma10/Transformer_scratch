def get_assessments_by_search(self, assessment_query, assessment_search):
        """Pass through to provider AssessmentSearchSession.get_assessments_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_assessments_by_search(assessment_query, assessment_search)