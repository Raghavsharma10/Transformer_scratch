def set_blend_func(self, srgb='one', drgb='zero',
                       salpha=None, dalpha=None):
        """Specify pixel arithmetic for RGB and alpha
    
        Parameters
        ----------
        srgb : str
            Source RGB factor.
        drgb : str
            Destination RGB factor.
        salpha : str | None
            Source alpha factor. If None, ``srgb`` is used.
        dalpha : str
            Destination alpha factor. If None, ``drgb`` is used.
        """
        salpha = srgb if salpha is None else salpha
        dalpha = drgb if dalpha is None else dalpha
        self.glir.command('FUNC', 'glBlendFuncSeparate', 
                          srgb, drgb, salpha, dalpha)