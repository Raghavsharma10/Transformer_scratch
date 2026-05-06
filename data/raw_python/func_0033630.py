def load_path(self, path):
        '''
        Load and return a given import path to a module or class
        '''
        containing_module, _, last_item = path.rpartition('.')
        if last_item[0].isupper():
            # Is a class definition, should do an "import from"
            path = containing_module
        imported_obj = importlib.import_module(path)
        if last_item[0].isupper():
            try:
                imported_obj = getattr(imported_obj, last_item)
            except AttributeError:
                msg = 'Cannot import "%s". ' \
                    '(Hint: CamelCase is only for classes)' % last_item
                raise ConfigurationError(msg)
        return imported_obj