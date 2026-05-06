def accept(self, data):
        '''
        Try to accpet MultiDict-like object and return if it is valid.
        '''
        self.raw_data = MultiDict(data)
        self.errors = {}
        for field in self.fields:
            if field.writable:
                self.python_data.update(field.accept())
            else:
                for name in field.field_names:
                    # readonly field
                    subfield = self.get_field(name)
                    value = self.python_data[subfield.name]
                    subfield.set_raw_value(self.raw_data, subfield.from_python(value))
        return self.is_valid