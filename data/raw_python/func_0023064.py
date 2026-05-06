def get_state(self):
        """ Get the current view state of the camera

        Returns a dict of key-value pairs. The exact keys depend on the
        camera. Can be passed to set_state() (of this or another camera
        of the same type) to reproduce the state.
        """
        D = {}
        for key in self._state_props:
            D[key] = getattr(self, key)
        return D