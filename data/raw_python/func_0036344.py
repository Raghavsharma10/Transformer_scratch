def load_servers_from_env(self, filter=[], dynamic=None):
        '''Load the name servers environment variable and parse each server in
        the list.

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
        if NAMESERVERS_ENV_VAR in os.environ:
            servers = [s for s in os.environ[NAMESERVERS_ENV_VAR].split(';') \
                         if s]
            self._parse_name_servers(servers, filter, dynamic)