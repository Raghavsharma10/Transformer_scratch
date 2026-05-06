def joinOn(self, model, onIndex):
        """
        Performs an eqJoin on with the given model. The resulting join will be
        accessible through the models name.
        """
        return self._joinOnAsPriv(model, onIndex, model.__name__)