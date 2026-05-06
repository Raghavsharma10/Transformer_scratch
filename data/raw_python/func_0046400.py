def get_actual_start_time(self):
        """Gets the time this assessment was started.

        return: (osid.calendaring.DateTime) - the start time
        raise:  IllegalState - ``has_started()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        if not self.has_started():
            raise errors.IllegalState('this assessment has not yet started')
        if self._my_map['actualStartTime'] is None:
            raise errors.IllegalState('this assessment has not yet been started by the taker')
        else:
            start_time = self._my_map['actualStartTime']
            return DateTime(year=start_time.year,
                            month=start_time.month,
                            day=start_time.day,
                            hour=start_time.hour,
                            minute=start_time.minute,
                            second=start_time.second,
                            microsecond=start_time.microsecond)