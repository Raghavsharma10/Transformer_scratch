def _compile_lock(self, query, value):
        """
        Compile the lock into SQL

        :param query: A QueryBuilder instance
        :type query: QueryBuilder

        :param value: The lock value
        :type value: bool or str

        :return: The compiled lock
        :rtype: str
        """
        if isinstance(value, basestring):
            return value

        if value is True:
            return 'FOR UPDATE'
        elif value is False:
            return 'LOCK IN SHARE MODE'