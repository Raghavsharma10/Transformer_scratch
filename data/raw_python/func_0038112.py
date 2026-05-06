def is_decimal(self):
        """Determine if a data record is of the type float."""
        dt = DATA_TYPES['decimal']
        if type(self.data) in dt['type']:
            self.type = 'DECIMAL'
            num_split = str(self.data).split('.', 1)
            self.len = len(num_split[0])
            self.len_decimal = len(num_split[1])
            return True