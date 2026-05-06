def _dyn_loader(self, module: str, kwargs: str):
        """Dynamically load a specific module instance."""
        package_directory: str = os.path.dirname(os.path.abspath(__file__))
        modules: str = package_directory + "/modules"
        module = module + ".py"
        if module not in os.listdir(modules):
            raise Exception("Module %s is not valid" % module)
        module_name: str = module[:-3]
        import_path: str = "%s.%s" % (self.MODULE_PATH, module_name)
        imported = import_module(import_path)
        obj = getattr(imported, 'Module')
        return obj(**kwargs)