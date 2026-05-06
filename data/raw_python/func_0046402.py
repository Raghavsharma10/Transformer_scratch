def get_completion_time(self):
        """Gets the time of this assessment was completed.

        return: (osid.calendaring.DateTime) - the end time
        raise:  IllegalState - ``has_ended()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        if not self.has_ended():
            raise errors.IllegalState('this assessment has not yet ended')
        if not self._my_map['completionTime']:
            raise errors.OperationFailed('someone forgot to set the completion time')
        completion_time = self._my_map['completionTime']
        return DateTime(year=completion_time.year,
                        month=completion_time.month,
                        day=completion_time.day,
                        hour=completion_time.hour,
                        minute=completion_time.minute,
                        second=completion_time.second,
                        microsecond=completion_time.microsecond)