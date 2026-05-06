def module(self):
        """The module specified by the ``library`` attribute."""

        if self._module is None:
            if self.library is None:
                raise ValueError(
                    "Backend '%s' doesn't specify a library attribute" % self.__class__)

            try:
                if '.' in self.library:
                    mod_path, cls_name = self.library.rsplit('.', 1)
                    mod = import_module(mod_path)
                    self._module = getattr(mod, cls_name)
                else:
                    self._module = import_module(self.library)
            except (AttributeError, ImportError):
                raise ValueError("Couldn't load %s backend library" % cls_name)

        return self._module