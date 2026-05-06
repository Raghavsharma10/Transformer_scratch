def get_duration(self):
        """Gets the duration for this assessment.

        return: (osid.calendaring.Duration) - the duration
        raise:  IllegalState - ``has_duration()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.assessment.AssessmentOffered.get_duration_template
        if not bool(self._my_map['duration']):
            raise errors.IllegalState()
        return Duration(**self._my_map['duration'])