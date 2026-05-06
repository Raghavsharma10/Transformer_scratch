def format(self, template=None):
        """ Substitutes variables within template with that of fields'
        """
        pattern = r"(?:<([^<]*?)\$(\w+)([^>]*?)>)"
        s = sub(pattern, self._format_repl, template or self.template)
        s = self._str_fix_whitespace(s)
        return s