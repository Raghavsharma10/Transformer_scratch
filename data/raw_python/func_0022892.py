def set_hint(self, target, mode):
        """Set OpenGL drawing hint
    
        Parameters
        ----------
        target : str
            The target, e.g. 'fog_hint', 'line_smooth_hint',
            'point_smooth_hint'.
        mode : str
            The mode to set (e.g., 'fastest', 'nicest', 'dont_care').
        """
        if not all(isinstance(tm, string_types) for tm in (target, mode)):
            raise TypeError('target and mode must both be strings')
        self.glir.command('FUNC', 'glHint', target, mode)