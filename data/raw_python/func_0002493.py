def associate(self, model):
        """
        Associate the model instance to the given parent.

        :type model: eloquent.Model

        :rtype: eloquent.Model
        """
        self._parent.set_attribute(self._foreign_key, model.get_key())
        self._parent.set_attribute(self._morph_type, model.get_morph_class())

        return self._parent.set_relation(self._relation, model)