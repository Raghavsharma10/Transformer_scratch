def set_gl_state(self, preset=None, **kwargs):
        """Define the set of GL state parameters to use when drawing

        Parameters
        ----------
        preset : str
            Preset to use.
        **kwargs : dict
            Keyword arguments to `gloo.set_state`.
        """
        self._vshare.gl_state = kwargs
        self._vshare.gl_state['preset'] = preset