def mu(self):
        """See docs for `Model` abstract base class."""
        mu = self._models[0].mu
        assert all([mu == model.mu for model in self._models])
        return mu