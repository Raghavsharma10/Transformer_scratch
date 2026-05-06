def tiny_integer(self, column, auto_increment=False, unsigned=False):
        """
        Create a new tiny integer column on the table.

        :param column: The column
        :type column: str

        :type auto_increment: bool

        :type unsigned: bool

        :rtype: Fluent
        """
        return self._add_column('tiny_integer', column,
                                auto_increment=auto_increment,
                                unsigned=unsigned)