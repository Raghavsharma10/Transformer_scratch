def _render_picking(self, **kwargs):
        """Render the scene in picking mode, returning a 2D array of visual 
        IDs.
        """
        try:
            self._scene.picking = True
            img = self.render(bgcolor=(0, 0, 0, 0), **kwargs)
        finally:
            self._scene.picking = False
        img = img.astype('int32') * [2**0, 2**8, 2**16, 2**24]
        id_ = img.sum(axis=2).astype('int32')
        return id_