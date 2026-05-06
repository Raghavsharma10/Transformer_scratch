def wrapped_setups(self):
        """Create tokens for Described.setup = noy_wrap_setup(Described, Described.setup) for setup/teardown"""
        lst = []
        for group in self.all_groups:
            if not group.root:
                if group.has_after_each:
                    lst.extend(self.tokens.wrap_after_each(group.kls_name, group.async_after_each))

                if group.has_before_each:
                    lst.extend(self.tokens.wrap_before_each(group.kls_name, group.async_before_each))

        if lst:
            indentation_reset = [
                  (NEWLINE, '\n')
                , (INDENT, '')
                ]
            lst = indentation_reset + lst

        return lst