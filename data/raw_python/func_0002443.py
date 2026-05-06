def dissociate(self):
        """
        Dissociate previously associated model from the given parent.

        :rtype: eloquent.Model
        """
        self._parent.set_attribute(self._foreign_key, None)

        return self._parent.set_relation(self._relation, None)