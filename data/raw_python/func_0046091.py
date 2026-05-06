def get_time_as_string(self):
        """stub"""
        if self.has_time():
            return (str(self.time['hours']).zfill(2) + ':' +
                    str(self.time['minutes']).zfill(2) + ':' +
                    str(self.time['seconds']).zfill(2))
        raise IllegalState()