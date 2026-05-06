def import_path_to_absolute_path(import_path, file_marker):
        '''
        Given a Python import path, convert to a likely absolute filesystem
        path, by searching for the given filename marker (such as
        'package.json' or '__init__.py') through the Python system path. Do not
        return given filename.
        '''
        path_fragment = import_path.replace('.', os.path.sep)
        path_suffix = os.path.join(path_fragment, file_marker)
        for path_base in sys.path:
            path = os.path.join(path_base, path_suffix)
            if os.path.exists(path):
                return os.path.join(path_base, path_fragment)
        msg = 'Cannot find import path: %s, %s'
        raise ConfigurationError(msg % (import_path, file_marker))