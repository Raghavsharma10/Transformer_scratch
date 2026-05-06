def on_key_release_repeat(self, *dummy):
        """
        Avoid repeated trigger of callback.

        When holding a key down, multiple key press and release events
        are fired in succession. Debouncing is implemented to squash these.
        """
        self.has_prev_key_release = self.after_idle(self.on_key_release, dummy)