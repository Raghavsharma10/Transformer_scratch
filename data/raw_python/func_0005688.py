def add_success(self, group=None, type_='', field='', description=''):
        """parse and append a success data param"""
        group = group or '(200)'
        group = int(group.lower()[1:-1])
        self.retcode = self.retcode or group
        if group != self.retcode:
            raise ValueError('Two or more retcodes!')
        type_ = type_ or '{String}'
        p = Param(type_, field, description)
        self.params['responce'][p.field] = p