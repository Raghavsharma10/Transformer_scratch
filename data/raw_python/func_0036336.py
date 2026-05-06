def add_name_server(self, server, filter=[], dynamic=None):
        '''Parse a name server, adding its contents to the tree.

        @param server The address of the name server, in standard
                      address format. e.g. 'localhost',
                      'localhost:2809', '59.7.0.1'.
        @param filter Restrict the parsed objects to only those in this
                      path. For example, setting filter to [['/',
                      'localhost', 'host.cxt', 'comp1.rtc']] will
                      prevent 'comp2.rtc' in the same naming context
                      from being parsed.
        @param dynamic Override the tree-wide dynamic setting. If not provided,
                       the value given when the tree was created will be used.

        '''
        if dynamic == None:
            dynamic = self._dynamic
        self._parse_name_server(server, filter, dynamic=dynamic)