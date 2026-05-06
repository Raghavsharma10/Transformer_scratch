def associate(self, model):
        """
        Associate the model instance to the given parent.

        :type model: eloquent.Model

        :rtype: eloquent.Model
        """
        self._parent.set_attribute(self._foreign_key, model.get_attribute(self._other_key))

        return self._parent.set_relation(self._relation, model)