def fill_with_model(self, model, *fields):
        """
        Populates this form with localized properties from model.
        :param fields: string list indicating the fields to include. If None, all fields defined on form will be used
        :param model: model
        :return: dict with localized properties
        """
        model_dct = model.to_dict(include=self._fields.keys())
        localized_dct = self.localize(*fields, **model_dct)
        if model.key:
            localized_dct['id'] = model.key.id()
        return localized_dct