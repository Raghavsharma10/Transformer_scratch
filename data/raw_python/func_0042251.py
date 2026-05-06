def form_invalid(self, form):
        '''Builds the JSON for the errors'''
        response = {self.errors_key: {}}
        response[self.non_field_errors_key] = form.non_field_errors()
        response.update(self.get_hidden_fields_errors(form))

        for field in form.visible_fields():
            if field.errors:
                response[self.errors_key][field.html_name] = self._get_field_error_dict(field)

        if self.include_success:
            response[self.sucess_key] = False

        return self._render_json(response)