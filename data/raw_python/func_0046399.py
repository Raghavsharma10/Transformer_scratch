def has_started(self):
        """Tests if this assessment has begun.

        return: (boolean) - ``true`` if the assessment has begun,
                ``false`` otherwise
        *compliance: mandatory -- This method must be implemented.*

        """
        assessment_offered = self.get_assessment_offered()
        if assessment_offered.has_start_time():
            return DateTime.utcnow() >= assessment_offered.get_start_time()
        else:
            return True