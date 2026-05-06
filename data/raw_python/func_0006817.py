def basic_types(self):
        """Returns non-postgres types referenced in user supplied model """
        if not self.foreign_key_definitions:
            return self.standard_types
        else:
            tmp = self.standard_types
            tmp.append('ForeignKey')
            return tmp