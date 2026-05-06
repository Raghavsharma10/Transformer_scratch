def set_key_state(self, key, state):
        """Sets the key state and redraws it.

        :param key: Key to update state for.
        :param state: New key state.
        """
        key.state = state
        self.renderer.draw_key(self.surface, key)