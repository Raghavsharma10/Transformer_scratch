def dependencies(self, kwargs=None, expand_only=False):
        """Returns all dependencies of this assistant with regards to specified kwargs.

        If expand_only == False, this method returns list of mappings of dependency types
        to actual dependencies (keeps order, types can repeat), e.g.
        Example:
        [{'rpm', ['rubygems']}, {'gem', ['mygem']}, {'rpm', ['spam']}, ...]
        If expand_only == True, this method returns a structure that can be used as
        "dependencies" section and has all the "use: foo" commands expanded (but conditions
        are left untouched and variables are not substituted).
        """
        # we can't use {} as a default for kwargs, as that initializes the dict only once in Python
        # and uses the same dict in all subsequent calls of this method
        if not kwargs:
            kwargs = {}

        self.proper_kwargs('dependencies', kwargs)
        sections = self._get_dependency_sections_to_use(kwargs)
        deps = []

        for sect in sections:
            if expand_only:
                deps.extend(lang.expand_dependencies_section(sect, kwargs))
            else:
                deps.extend(lang.dependencies_section(sect, kwargs, runner=self))

        return deps