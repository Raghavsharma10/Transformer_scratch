def load_commands_from_entry_point(self, specifier):
        """
        Load commands defined within a pkg_resources entry point.

        Each entry will be a module that should be searched for functions
        decorated with the :func:`subparse.command` decorator. This
        operation is not recursive.

        """
        for ep in pkg_resources.iter_entry_points(specifier):
            module = ep.load()
            command.discover_and_call(module, self.command)