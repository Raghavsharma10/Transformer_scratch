def count(self, query=None, date=None, table=None):
        '''
        Run a query on the given cube and return only
        the count of resulting matches.

        :param query: The query in pql
        :param date: date (metrique date range) that should be queried
                    If date==None then the most recent versions of the
                    objects will be queried.
        :param collection: cube name
        :param owner: username of cube owner
        '''
        table = table or self.config.get('table')
        sql_count = select([func.count()])
        query = self._parse_query(table=table, query=query, date=date,
                                  fields='id', alias='anon_x')

        if query is not None:
            query = sql_count.select_from(query)
        else:
            table = self.get_table(table)
            query = sql_count
            query = query.select_from(table)
        return self.session_auto.execute(query).scalar()