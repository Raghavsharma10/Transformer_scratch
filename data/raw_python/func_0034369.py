def _joinOnAsPriv(self, model, onIndex, whatAs):
        """
        Private method for handling joins.
        """
        if self._join:
            raise Exception("Already joined with a table!")

        self._join = model
        self._joinedField = whatAs
        table = model.table
        self._query = self._query.eq_join(onIndex, r.table(table))
        return self