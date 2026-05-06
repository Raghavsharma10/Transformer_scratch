def search(self, *query, **kwargs):
        '''
        Searches based on tags specified by users



        Parameters
        ---------
        query: str
            tags to search on. If multiple terms, provided in comma delimited
            string format

        prefix: str
            start of archive name. Providing a start string improves search
            speed.

        '''

        prefix = kwargs.get('prefix')

        if prefix is not None:
            prefix = fs.path.relpath(prefix)

        return self.manager.search(query, begins_with=prefix)