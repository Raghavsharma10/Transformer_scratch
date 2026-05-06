def get_assessment_part(self):
        """If there's an AssessmentSection ask it first for the part.

        This will take advantage of the fact that the AssessmentSection may
        have already cached the Part in question.

        """
        if self._magic_parent_id is None:
            assessment_part_id = Id(self.my_osid_object._my_map['assessmentPartId'])
        else:
            assessment_part_id = self._magic_parent_id
        if self._assessment_section is not None:
            return self._assessment_section._get_assessment_part(assessment_part_id)
        # else:
        apls = get_assessment_part_lookup_session(runtime=self.my_osid_object._runtime,
                                                  proxy=self.my_osid_object._proxy,
                                                  section=self._assessment_section)
        apls.use_federated_bank_view()
        apls.use_unsequestered_assessment_part_view()
        return apls.get_assessment_part(assessment_part_id)