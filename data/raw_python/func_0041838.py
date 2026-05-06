def validate_args(self):
        """Input validators for this rule type."""
        base.BaseTarget.validate_args(self)
        params = self.params
        if params['extra_control_fields'] is not None:
            assert isinstance(params['extra_control_fields'], list), (
                'extra_control_fields must be a list of tuples, not %s' % type(
                    params['extra_control_fields']))
            for elem in params['extra_control_fields']:
                assert (isinstance(elem, tuple) and len(elem) == 1), (
                    'extra_control_fields must be a list of 2-element tuples. '
                    'Invalid contents: %s' % elem)
        pkgname_re = '^[a-z][a-z0-9+-.]+'
        assert re.match(pkgname_re, params['package_name']), (
            'Invalid package name: %s. Must match %s' % (
                params['package_name'], pkgname_re))