def set_doc(self, doc: str):
        """Assign the given docstring to the property instance and, if
        possible, to the `__test__` dictionary of the module of its
        owner class."""
        self.__doc__ = doc
        if hasattr(self, 'module'):
            ref = f'{self.objtype.__name__}.{self.name}'
            self.module.__dict__['__test__'][ref] = doc