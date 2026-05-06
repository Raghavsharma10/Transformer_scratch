def render_source(self, source, variables=None):
        """
        Render a source with the passed variables.
        """
        if variables is None:
            variables = {}
        template = self._engine.from_string(source)
        return template.render(**variables)