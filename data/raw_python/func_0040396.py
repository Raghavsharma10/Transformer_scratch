def exec_module(self, module):
        """Execute the module using the old imp."""
        path = [os.path.dirname(module.__file__)]  # file should have been resolved before (module creation)
        file = None
        try:
            file, pathname, description = imp.find_module(module.__name__.rpartition('.')[-1], path)
            module = imp.load_module(module.__name__, file, pathname, description)
        finally:
            if file:
                file.close()