def _dep_changed(self, dep, code_changed=False, value_changed=False):
        """ Called when a dependency's expression has changed.
        """
        self.changed(code_changed, value_changed)