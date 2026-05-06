def get_transform(self, map_from='visual', map_to='render'):
        """Return a transform mapping between any two coordinate systems.
        
        Parameters
        ----------
        map_from : str
            The starting coordinate system to map from. Must be one of: visual,
            scene, document, canvas, framebuffer, or render.
        map_to : str
            The ending coordinate system to map to. Must be one of: visual,
            scene, document, canvas, framebuffer, or render.
        """
        tr = ['visual', 'scene', 'document', 'canvas', 'framebuffer', 'render']
        ifrom = tr.index(map_from)
        ito = tr.index(map_to)
        
        if ifrom < ito:
            trs = [getattr(self, '_' + t + '_transform')
                   for t in tr[ifrom:ito]][::-1]
        else:
            trs = [getattr(self, '_' + t + '_transform').inverse
                   for t in tr[ito:ifrom]]
        return self._cache.get(trs)