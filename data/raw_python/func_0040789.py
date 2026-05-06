def validate_field(self, value):
        '''
        Method that must validate the value
        It must return None if the value is valid and a error msg otherelse.
        Ex: If expected input must be int, validate should a return a msg like
        "The filed must be a integer value"
        '''
        if self.choices:
            value = self.normalize_field(value)
            if value in self.choices:
                return None
            return _('Must be one of: %(choices)s') % {'choices': '; '.join(self.choices)}
        if self.default is not None:
            if value is None or value == '':
                value = self.default
        if self.required and (value is None or value == ''):
            return _('Required field')