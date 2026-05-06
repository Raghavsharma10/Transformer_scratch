def _date_based_where(self, type, query, where):
        """
        Compiled a date where based clause

        :param type: The date type
        :type type: str

        :param query: A QueryBuilder instance
        :type query: QueryBuilder

        :param where: The condition
        :type where: dict

        :return: The compiled clause
        :rtype: str
        """
        value = str(where['value']).zfill(2)
        value = self.parameter(value)

        return 'strftime(\'%s\', %s) %s %s'\
               % (type, self.wrap(where['column']),
                  where['operator'], value)