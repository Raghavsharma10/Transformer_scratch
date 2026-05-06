def get_end_date(self):
        """Gets the end date.

        return: (osid.calendaring.DateTime) - the end date
        *compliance: mandatory -- This method must be implemented.*

        """
        edate = self.my_osid_object._my_map['endDate']
        return DateTime(
            edate.year,
            edate.month,
            edate.day,
            edate.hour,
            edate.minute,
            edate.second,
            edate.microsecond)