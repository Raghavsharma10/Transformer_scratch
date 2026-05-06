def changed(self, code_changed=False, value_changed=False):
        """Inform dependents that this shaderobject has changed.
        """
        for d in self._dependents:
            d._dep_changed(self, code_changed=code_changed,
                           value_changed=value_changed)