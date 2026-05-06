def get_start_time(self):
        """Gets the start time for this assessment.

        return: (osid.calendaring.DateTime) - the designated start time
        raise:  IllegalState - ``has_start_time()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.assessment.AssessmentOffered.get_start_time_template
        if not bool(self._my_map['startTime']):
            raise errors.IllegalState()
        dt = self._my_map['startTime']
        return DateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)