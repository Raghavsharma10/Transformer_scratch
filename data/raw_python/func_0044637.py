def generate(self):
        """
        Generate Lambda function source; return it as a string.

        :rtype: str
        :returns: lambda function source
        """
        s = self._docstring
        s += self._get_source().replace(
            'endpoints = {}',
            'endpoints = ' + self._config_src
        ).replace(
            'logger.setLevel(logging.INFO)',
            'logger.setLevel(logging.%s)' % self.config.logging_level
        )
        return s