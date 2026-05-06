def set_fallback(self, target):
        """Sets a fallback configuration for section.

        Re-exec uWSGI with the specified config when exit code is 1.

        :param str|unicode|Section target: File path or Section to include.
        """
        if isinstance(target, Section):
            target = ':' + target.name

        self._set('fallback-config', target)

        return self