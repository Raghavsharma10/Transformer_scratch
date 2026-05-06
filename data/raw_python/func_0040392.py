def load_module(self, fullname):
        """Load the specified module into sys.modules and return it.
        This method is for python2 only, but implemented with backported py3 methods.
        """

        if fullname in sys.modules:
            mod = sys.modules[fullname]
            self.exec_module(mod)
            # In this case we do not want to remove the module in case of error
            # Ref : https://docs.python.org/3/reference/import.html#loaders
        else:
            try:
                # Retrieving the spec to help creating module properly
                spec = spec_from_loader(fullname, self)

                # this will call create_module and also initialize the module properly (like for py3)
                mod = module_from_spec(spec)

                # as per https://docs.python.org/3/reference/import.html#loaders
                assert mod.__name__ in sys.modules

                self.exec_module(mod)
                # We don't ensure that the import-related module attributes get
                # set in the sys.modules replacement case.  Such modules are on
                # their own.
            except Exception as exc:
                # TODO : log exception !
                # as per https://docs.python.org/3/reference/import.html#loaders
                if fullname in sys.modules:
                    del sys.modules[fullname]
                raise

        return sys.modules[fullname]