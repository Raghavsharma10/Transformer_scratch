def add_model(self, allow_alias=False, **kwargs):
        """Add a `Model` instance to this entry."""
        if not allow_alias and MODEL.ALIAS in kwargs:
            err_str = "`{}` passed in kwargs, this shouldn't happen!".format(
                SOURCE.ALIAS)
            self._log.error(err_str)
            raise RuntimeError(err_str)

        # Set alias number to be +1 of current number of models
        if MODEL.ALIAS not in kwargs:
            kwargs[MODEL.ALIAS] = str(self.num_models() + 1)
        model_obj = self._init_cat_dict(Model, self._KEYS.MODELS, **kwargs)
        if model_obj is None:
            return None

        for item in self.get(self._KEYS.MODELS, ''):
            if model_obj.is_duplicate_of(item):
                return item[item._KEYS.ALIAS]

        self.setdefault(self._KEYS.MODELS, []).append(model_obj)
        return model_obj[model_obj._KEYS.ALIAS]