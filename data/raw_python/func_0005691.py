def render_docstring(self):
        """make a nice docstring for ipython"""
        default = (' = ' + str(self.default)) if self.default else ''
        opt = 'optional' if self.is_optional else ''
        can_be = ' '.join(self.possible_values) if self.possible_values else ''
        can_be = 'one of [{}]'.format(can_be) if can_be else ''
        type_ = 'of type "' + str(self.type) + '"'
        res = ' '.join([opt, '"' + self.field + '"', default, type_, can_be, '\n'])
        return res.replace('  ', ' ').lstrip()