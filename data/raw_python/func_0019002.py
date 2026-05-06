def _init_methods(self):
        """Convert all pure Python calculation functions of the model class to
        methods and assign them to the model instance.
        """
        for name_group in self._METHOD_GROUPS:
            functions = getattr(self, name_group, ())
            uniques = {}
            for func in functions:
                name_func = func.__name__
                method = types.MethodType(func, self)
                setattr(self, name_func, method)
                shortname = '_'.join(name_func.split('_')[:-1])
                if shortname in uniques:
                    uniques[shortname] = None
                else:
                    uniques[shortname] = method
            for (shortname, method) in uniques.items():
                if method is not None:
                    setattr(self, shortname, method)