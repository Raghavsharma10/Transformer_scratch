def save(self, model, joining=None, touch=True):
        """
        Save a new model and attach it to the parent model.

        :type model: eloquent.Model
        :type joining: dict
        :type touch: bool

        :rtype: eloquent.Model
        """
        if joining is None:
            joining = {}

        model.save({'touch': False})

        self.attach(model.get_key(), joining, touch)

        return model