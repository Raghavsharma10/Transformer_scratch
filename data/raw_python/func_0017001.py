def gitignore_templates(self):
        """Returns the list of available templates.

        :returns: list of template names
        """
        url = self._build_url('gitignore', 'templates')
        return self._json(self._get(url), 200) or []