def get_data(self, compact=True):
        '''
        Returns data representing current state of the form. While
        Form.raw_data may contain alien fields and invalid data, this method
        returns only valid fields that belong to this form only. It's designed
        to pass somewhere current state of the form (as query string or by
        other means).
        '''
        data = MultiDict()
        for field in self.fields:
            raw_value = field.from_python(self.python_data[field.name])
            field.set_raw_value(data, raw_value)
        if compact:
            data = MultiDict([(k, v) for k, v in data.items() if v])
        return data