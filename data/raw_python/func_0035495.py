def updateParams(self, newvalues, update_all=False):
        """See docs for `Model` abstract base class."""
        assert all(map(lambda x: x in self.freeparams, newvalues.keys())),\
                "Invalid entry in newvalues: {0}\nfreeparams: {1}".format(
                ', '.join(newvalues.keys()), ', '.join(self.freeparams))
        changed = set([]) # contains string names of changed params
        for (name, value) in newvalues.items():
            _checkParam(name, value, self.PARAMLIMITS, self.PARAMTYPES)
            if isinstance(value, scipy.ndarray):
                if (value != getattr(self, name)).any():
                    changed.add(name)
                    setattr(self, name, value.copy())
            else:
                if value != getattr(self, name):
                    changed.add(name)
                    setattr(self, name, copy.copy(value))

        if update_all or changed:
            self._cached = {}

        # The order of the updating below is important.
        # If you change it, you may break either this class
        # **or** classes that inherit from it.
        # Note also that not all attributes need to be updated
        # for all possible parameter changes, but just doing it
        # this way is much simpler and adds negligible cost.
        if update_all or (changed and changed != set(['mu'])):
            self._update_pi_vars()
            self._update_phi()
            self._update_prx()
            self._update_dprx()
            self._update_Qxy()
            self._update_Frxy()
            self._update_Prxy()
            self._update_Prxy_diag()
            self._update_dPrxy()
            self._update_B()