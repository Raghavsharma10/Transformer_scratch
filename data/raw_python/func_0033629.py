def load_all(self, key, default=None):
        '''
        Import settings key as a dict or list with values of importable paths
        If a default constructor is specified, and a path is not importable, it
        falls back to running the given constructor.
        '''
        value = getattr(self, key)
        if default is not None:
            def loader(path): return self.load_path_with_default(path, default)
        else:
            loader = self.load_path
        if isinstance(value, dict):
            return {key: loader(value) for key, value in value.items()}
        elif isinstance(value, list):
            return [loader(value) for value in value]
        else:
            raise ValueError('load_all must be list or dict')