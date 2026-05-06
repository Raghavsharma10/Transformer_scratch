def source_files(self):
        """This rule's source files."""
        if 'srcs' in self.params and self.params['srcs'] is not None:
            return util.flatten(self.params['srcs'])