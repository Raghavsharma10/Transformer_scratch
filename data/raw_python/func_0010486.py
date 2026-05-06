def _module_name(self) -> str:
        """Module name of the wrapped function."""
        name = self.f.__module__
        if name == '__main__':
            return importer.main_module_name()
        return name