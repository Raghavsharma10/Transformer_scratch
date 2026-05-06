def validate_args(self):
        """Input validation!"""
        def validate_name():
            allowed_re = '^[a-z](([a-z0-9_-]+)?([a-z0-9])?)?'
            assert isinstance(self.params['name'], basestring), (
                'Name must be a string, not %s' % repr(self.params['name']))
            assert re.match(allowed_re, self.params['name']), (
                'Invalid rule name: %s. Must match %s.' % (
                    repr(self.params['name']), repr(allowed_re)))
        validate_name()

        def validate_deps():
            if 'deps' in self.params:
                assert type(self.params['deps']) in (type(None), list), (
                    'Deps must be a list, not %s' % repr(self.params['deps']))
        validate_deps()