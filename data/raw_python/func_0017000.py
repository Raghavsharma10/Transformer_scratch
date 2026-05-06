def gitignore_template(self, language):
        """Returns the template for language.

        :returns: str
        """
        url = self._build_url('gitignore', 'templates', language)
        json = self._json(self._get(url), 200)
        if not json:
            return ''
        return json.get('source', '')