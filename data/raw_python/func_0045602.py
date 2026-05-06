def release(self):
        """Destroys the state, along with its functions."""
        self.clear()

        if hasattr(self, "functions"):
            del self.functions

        if hasattr(self, "lib") and self.lib is not None:
            self.lib._jit_destroy_state(self.state)
            self.lib = None