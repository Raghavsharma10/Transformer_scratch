def _release(self):
        """Destroy self since closures cannot be called again."""
        del self.funcs
        del self.variables
        del self.variable_values
        del self.satisfied