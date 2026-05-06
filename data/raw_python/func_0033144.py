def count(self, query=None, date=None):
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
        return self.proxy.count(table=self.name, query=query, date=date)