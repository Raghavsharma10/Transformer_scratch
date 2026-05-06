def creation_date(self):
        """ Returns the account creation date as a localtime time.struct_time
        struct if public"""
        timestamp = self._prof.get("timecreated")
        if timestamp:
            return time.localtime(timestamp)