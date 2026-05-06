def create(self, data={}, store=None):
        """Initiazes an OPR

        First step in the OPR process is to create the OPR request.
        Returns the OPR token
        """
        _store = store or self.store
        _data = self._build_opr_data(data, _store) if data else self._opr_data
        return self._process('opr/create', _data)