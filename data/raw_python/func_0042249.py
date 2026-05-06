def _get_field_error_dict(self, field):
        '''Returns the dict containing the field errors information'''
        return {
            'name': field.html_name,
            'id': 'id_{}'.format(field.html_name), # This may be a problem
            'errors': field.errors,
        }