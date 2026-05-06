def to_dict(self):
        """
        Return this ModuleDoc as a dict.  In addition to `CommentDoc` defaults,
        this has:

            - **name**: The module name.
            - **dependencies**: A list of immediate dependencies.
            - **all_dependencies**: A list of all dependencies.
        """
        vars = super(ModuleDoc, self).to_dict()
        vars['dependencies'] = self.dependencies
        vars['name'] = self.name
        try:
            vars['all_dependencies'] = self.all_dependencies[:]
        except AttributeError:
            vars['all_dependencies'] = []
        return vars