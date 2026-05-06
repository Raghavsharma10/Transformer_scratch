def use_sequestered_assessment_part_view(self):
        """Pass through to provider AssessmentPartLookupSession.use_sequestered_assessment_part_view"""
        # Does this need to be re-implemented to match the other non-sub-package view setters?
        self._containable_views['assessment_part'] = SEQUESTERED
        self._get_sub_package_provider_session('assessment_authoring',
                                               'assessment_part_lookup_session')
        for session in self._provider_sessions:
            for provider_session_name, provider_session in self._provider_sessions[session].items():
                try:
                    provider_session.use_sequestered_assessment_part_view()
                except AttributeError:
                    pass