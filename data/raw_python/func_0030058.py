def render(self, template_name, variables=None):
        """
        Render a template with the passed variables.
        """
        if variables is None:
            variables = {}
        template = self._engine.get_template(template_name)
        return template.render(**variables)