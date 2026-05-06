def _get_assessment_part(self, part_id=None):
        """Gets an AssessmentPart given a part_id.

        Returns this Section's own part if part_id is None.

        Make this a private part, so that it doesn't collide with the AssessmentPart.get_assessment_part
        method, which does not expect any arguments...

        """
        if part_id is None:
            return self._assessment_part
        if part_id not in self._assessment_parts:
            lookup_session = self._get_assessment_part_lookup_session()
            self._assessment_parts[part_id] = lookup_session.get_assessment_part(part_id)
        return self._assessment_parts[part_id]