def is_year(self):
        """Determine if a data record is of type YEAR."""
        dt = DATA_TYPES['year']
        if dt['min'] and dt['max']:
            if type(self.data) is dt['type'] and dt['min'] < self.data < dt['max']:
                self.type = 'year'.upper()
                self.len = None
                return True