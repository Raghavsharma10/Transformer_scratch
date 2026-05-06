def make_method_names(self):
        """Create tokens for setting __testname__ on functions"""
        lst = []
        for group in self.all_groups:
            for single in group.singles:
                name, english = single.name, single.english
                if english[1:-1] != name.replace('_', ' '):
                    lst.extend(self.tokens.make_name_modifier(not group.root, single.identifier, english))
        return lst