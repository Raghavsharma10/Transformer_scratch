def get_creation_time(self):
        """stub"""
        ct = self.my_osid_object._my_map['creationTime']
        return DateTime(ct.year,
                        ct.month,
                        ct.day,
                        ct.hour,
                        ct.minute,
                        ct.second,
                        ct.microsecond)