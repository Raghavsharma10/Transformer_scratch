def accept(self):
        '''
        Accepts all children fields, collects resulting values into dict and
        passes that dict to converter.

        Returns result of converter as separate value in parent `python_data`
        '''
        result = dict(self.python_data)
        for field in self.fields:
            if field.writable:
                result.update(field.accept())
            else:
                # readonly field
                field.set_raw_value(self.form.raw_data,
                                    field.from_python(result[field.name]))
        self.clean_value = self.conv.accept(result)
        return {self.name: self.clean_value}