def load(self, *modules):
        """Load one or more modules.

        Args:
            modules: Either a string full path to a module or an actual module
                object.
        """
        for module in modules:
            if isinstance(module, six.string_types):
                try:
                    module = get_object(module)
                except Exception as e:
                    self.errors[module] = e
                    continue
            self.modules[module.__package__] = module
            for (loader, module_name, is_pkg) in pkgutil.walk_packages(
                module.__path__
            ):
                full_name = "{}.{}".format(_package(module), module_name)
                try:
                    self.modules[full_name] = get_object(full_name)
                    if is_pkg:
                        self.load(self.modules[full_name])
                except Exception as e:
                    self.errors[full_name] = e