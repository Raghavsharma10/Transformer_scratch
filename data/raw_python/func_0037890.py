def is_varchar(self):
        """Determine if a data record is of the type VARCHAR."""
        dt = DATA_TYPES['varchar']
        if type(self.data) is dt['type'] and len(self.data) < dt['max']:
            self.type = 'VARCHAR'
            self.len = len(self.data)
            return True