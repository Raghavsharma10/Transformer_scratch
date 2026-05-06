def get_assessment_parts_by_search(self, assessment_part_query, assessment_part_search):
        """Pass through to provider AssessmentPartSearchSession.get_assessment_parts_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_assessment_parts_by_search(assessment_part_query, assessment_part_search)