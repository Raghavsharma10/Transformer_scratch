def accept(self):
        '''Extracts raw value from form's raw data and passes it to converter'''
        value = self.raw_value
        if not self._check_value_type(value):
            # XXX should this be silent or TypeError?
            value = [] if self.multiple else self._null_value
        self.clean_value = self.conv.accept(value)
        return {self.name: self.clean_value}