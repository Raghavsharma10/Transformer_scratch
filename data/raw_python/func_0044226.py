def compile_dependencies(self, sourcepath, include_self=False):
        """
        Apply compile on all dependencies

        Args:
            sourcepath (string): Sass source path to compile to its
                destination using project settings.

        Keyword Arguments:
            include_self (bool): If ``True`` the given sourcepath is add to
                items to compile, else only its dependencies are compiled.
        """
        items = self.inspector.parents(sourcepath)

        # Also add the current event related path
        if include_self:
            items.add(sourcepath)

        return filter(None, [self.compile_source(item) for item in items])