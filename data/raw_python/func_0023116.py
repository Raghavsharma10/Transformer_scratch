def set_gl_state(self, preset=None, **kwargs):
        """Define the set of GL state parameters to use when drawing

        Parameters
        ----------
        preset : str
            Preset to use.
        **kwargs : dict
            Keyword arguments to `gloo.set_state`.
        """
        for v in self._subvisuals:
            v.set_gl_state(preset=preset, **kwargs)