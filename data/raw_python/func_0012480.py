def _datetime_for_table_name(self, table_name):
        """
        Return a :py:class:`datetime.datetime` object for the date of the
        data in the specified table name.

        :param table_name: name of the table
        :type table_name: str
        :return: datetime that the table holds data for
        :rtype: datetime.datetime
        """
        m = self._table_re.match(table_name)
        dt = datetime.strptime(m.group(1), '%Y%m%d')
        return dt