def joinOnAs(self, model, onIndex, whatAs):
        """
        Like `joinOn` but allows setting the joined results name to access it
        from.

        Performs an eqJoin on with the given model. The resulting join will be
        accessible through the given name.
        """
        return self._joinOnAsPriv(model, onIndex, whatAs)