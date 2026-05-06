def fill_model(self, model=None):
        """
        Populates a model with normalized properties. If no model is provided (None) a new one will be created.
        :param model: model to be populade
        :return: populated model
        """
        normalized_dct = self.normalize()
        if model:
            if not isinstance(model, self._model_class):
                raise ModelFormSecurityError('%s should be %s instance' % (model, self._model_class.__name__))
            model.populate(**normalized_dct)
            return model
        return self._model_class(**normalized_dct)