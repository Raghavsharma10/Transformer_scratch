def make_describe_attrs(self):
        """Create tokens for setting is_noy_spec on describes"""
        lst = []
        if self.all_groups:
            lst.append((NEWLINE, '\n'))
            lst.append((INDENT, ''))

            for group in self.all_groups:
                if group.name:
                    lst.extend(self.tokens.make_describe_attr(group.kls_name))

        return lst