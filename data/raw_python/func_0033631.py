def load_path_with_default(self, path, default_constructor):
        '''
        Same as `load_path(path)', except uses default_constructor on import
        errors, or if loaded a auto-generated namespace package (e.g. bare
        directory).
        '''
        try:
            imported_obj = self.load_path(path)
        except (ImportError, ConfigurationError):
            imported_obj = default_constructor(path)
        else:
            # Ugly but seemingly expedient way to check a module was an
            # namespace type module
            if (isinstance(imported_obj, ModuleType) and
                    imported_obj.__spec__.origin == 'namespace'):
                imported_obj = default_constructor(path)
        return imported_obj