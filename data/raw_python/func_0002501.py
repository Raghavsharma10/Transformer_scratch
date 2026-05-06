def save(self, model):
        """
        Attach a model instance to the parent models.

        :param model: The model instance to attach
        :type model: Model

        :rtype: Model
        """
        model.set_attribute(self.get_plain_morph_type(), self._morph_class)

        return super(MorphOneOrMany, self).save(model)