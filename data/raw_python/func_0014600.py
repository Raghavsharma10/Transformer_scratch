def read(self, *args, **kwargs):
        '''Overridden read() method to call parse_flask_section() at the end'''
        ret = configparser.SafeConfigParser.read(self, *args, **kwargs)
        self.parse_flask_section()
        return ret