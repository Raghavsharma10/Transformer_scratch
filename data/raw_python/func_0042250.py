def get_hidden_fields_errors(self, form):
        '''Returns a dict to add in response when something is wrong with hidden fields'''
        if not self.include_hidden_fields or form.is_valid():
            return {}

        response = {self.hidden_field_error_key:{}}

        for field in form.hidden_fields():
            if field.errors:
                response[self.hidden_field_error_key][field.html_name] = self._get_field_error_dict(field)
        return response