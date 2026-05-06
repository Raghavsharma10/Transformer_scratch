def use(self, url, name='mytable'):
        '''Changes the data provider
        >>> yql.use('http://myserver.com/mytables.xml')
        '''
        self.yql_table_url = url
        self.yql_table_name = name
        return {'table url': url, 'table name': name}