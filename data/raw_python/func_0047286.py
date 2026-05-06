def _get_assessment_part_lookup_session(self):
        """need to account for magic parts"""
        section = getattr(self, '_assessment_section', None)
        session = get_assessment_part_lookup_session(self._runtime,
                                                     self._proxy,
                                                     section)
        session.use_unsequestered_assessment_part_view()
        session.use_federated_bank_view()
        return session