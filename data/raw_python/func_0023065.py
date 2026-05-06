def set_state(self, state=None, **kwargs):
        """ Set the view state of the camera

        Should be a dict (or kwargs) as returned by get_state. It can
        be an incomlete dict, in which case only the specified
        properties are set.

        Parameters
        ----------
        state : dict
            The camera state.
        **kwargs : dict
            Unused keyword arguments.
        """
        D = state or {}
        D.update(kwargs)
        for key, val in D.items():
            if key not in self._state_props:
                raise KeyError('Not a valid camera state property %r' % key)
            setattr(self, key, val)