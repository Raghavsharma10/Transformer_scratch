def ls(self, startswith=None):
        '''
        List all cubes available to the calling client.

        :param startswith: string to use in a simple "startswith" query filter
        :returns list: sorted list of cube names
        '''
        logger.info('Listing cubes starting with "%s")' % startswith)
        startswith = unicode(startswith or '')
        tables = sorted(name for name in self.db_tables
                        if name.startswith(startswith))
        return tables