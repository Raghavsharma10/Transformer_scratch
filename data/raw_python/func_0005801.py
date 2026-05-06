def add_expiration_rule(self, criterion, value, timeout, use_mod_time=False):
        """Adds statics expiration rule based on a criterion.

        :param str|unicode criterion: Criterion (subject) to base expiration on.

            See ``.expiration_criteria``.

        :param str|unicode|list[str|unicode] value: Value to test criteria upon.

            .. note:: Usually a regular expression.

        :param int timeout: Number of seconds to expire after.

        :param bool use_mod_time: Base on file modification time instead of the current time.

        """
        command = 'static-expires'
        separator = ' '

        if criterion != self.expiration_criteria.FILENAME:

            command += '-%s' % criterion

        if criterion == self.expiration_criteria.MIME_TYPE:

            separator = '='

        if use_mod_time:

            command += '-mtime'

        for value in listify(value):
            self._set(command, '%s%s%s' % (value, separator, timeout), multi=True)

        return self._section