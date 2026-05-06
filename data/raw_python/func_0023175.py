def _parse_template_vars(self):
        """ find all template variables in self._code, excluding the
        function name. 
        """
        template_vars = set()
        for var in parsing.find_template_variables(self._code):
            var = var.lstrip('$')
            if var == self.name:
                continue
            if var in ('pre', 'post'):
                raise ValueError('GLSL uses reserved template variable $%s' % 
                                 var)
            template_vars.add(var)
        return template_vars