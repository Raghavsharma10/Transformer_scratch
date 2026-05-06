def readfp(self, *args, **kwargs):
        '''Overridden readfp() method to call parse_flask_section() at the
        end'''
        ret = configparser.SafeConfigParser.readfp(self, *args, **kwargs)
        self.parse_flask_section()
        return ret