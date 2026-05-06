def desc(self, table):
        '''Returns table description
        >>> yql.desc('geo.countries')
        >>>
        '''
        query = "desc {0}".format(table)
        response = self.raw_query(query)

        return response