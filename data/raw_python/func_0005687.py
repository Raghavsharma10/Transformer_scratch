def add_param(self, group=None, type_='', field='', description=''):
        """parse and append a param"""
        group = group or '(Parameter)'
        group = group.lower()[1:-1]
        p = Param(type_, field, description)
        self.params[group][p.field] = p