def has_ended(self):
        """Tests if this assessment has ended.

        return: (boolean) - ``true`` if the assessment has ended,
                ``false`` otherwise
        *compliance: mandatory -- This method must be implemented.*

        """
        assessment_offered = self.get_assessment_offered()
        now = DateTime.utcnow()
        # There's got to be a better way to do this:
        if self._my_map['completionTime'] is not None:
            return True
        elif assessment_offered.has_deadline() and assessment_offered.has_duration():
            if self._my_map['actualStartTime'] is None:
                return now >= assessment_offered.get_deadline()
            else:
                return (now >= assessment_offered.get_deadline() and
                        now >= self._my_map['actualStartTime'] + assessment_offered.get_duration())
        elif assessment_offered.has_deadline():
            return now >= assessment_offered.get_deadline()
        elif assessment_offered.has_duration() and self._my_map['actualStartTime'] is not None:
            return now >= self._my_map['actualStartTime'] + assessment_offered.get_duration()
        else:
            return False