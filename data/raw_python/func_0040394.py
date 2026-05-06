def load_module(self, name):
        """Load a namespace module as if coming from an empty file.
        """
        _verbose_message('namespace module loaded with path {!r}', self.path)

        # Adjusting code from LoaderBasics
        if name in sys.modules:
            mod = sys.modules[name]
            self.exec_module(mod)
            # In this case we do not want to remove the module in case of error
            # Ref : https://docs.python.org/3/reference/import.html#loaders
        else:
            try:
                # Building custom spec and loading as in _LoaderBasics...
                spec = ModuleSpec(name, self, origin='namespace', is_package=True)
                spec.submodule_search_locations = self.path

                # this will call create_module and also initialize the module properly (like for py3)
                mod = module_from_spec(spec)

                # as per https://docs.python.org/3/reference/import.html#loaders
                assert mod.__name__ in sys.modules

                self.exec_module(mod)
                # We don't ensure that the import-related module attributes get
                # set in the sys.modules replacement case.  Such modules are on
                # their own.
            except:
                # as per https://docs.python.org/3/reference/import.html#loaders
                if name in sys.modules:
                    del sys.modules[name]
                raise

        return sys.modules[name]