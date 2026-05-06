def get_option(self, name, section=None, vars=None, expect=None):
        """Return an option from ``section`` with ``name``.

        :param section: section in the ini file to fetch the value; defaults to
        constructor's ``default_section``

        """
        vars = vars if vars else self.default_vars
        if section is None:
            section = self.default_section
        opts = self.get_options(section, opt_keys=[name], vars=vars)
        if opts:
            return opts[name]
        else:
            if self._narrow_expect(expect):
                raise ValueError('no option \'{}\' found in section {}'.
                                 format(name, section))