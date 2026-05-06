def get_time_spent(self):
        """Gets the total time spent taking this assessment.

        return: (osid.calendaring.Duration) - the total time spent
        *compliance: mandatory -- This method must be implemented.*

        """
        # Take another look at this. Not sure it's correct:
        if not self.has_started or not self.has_ended():
            raise errors.IllegalState()
        if self._my_map['completionTime'] is not None:
            return self.get_completion_time() - self.get_actual_start_time()
        else:
            raise errors.IllegalState()