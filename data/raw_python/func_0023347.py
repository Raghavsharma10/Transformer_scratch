def visuals_at(self, pos, radius=10):
        """Return a list of visuals within *radius* pixels of *pos*.
        
        Visuals are sorted by their proximity to *pos*.
        
        Parameters
        ----------
        pos : tuple
            (x, y) position at which to find visuals.
        radius : int
            Distance away from *pos* to search for visuals.
        """
        tr = self.transforms.get_transform('canvas', 'framebuffer')
        pos = tr.map(pos)[:2]

        id = self._render_picking(region=(pos[0]-radius, pos[1]-radius,
                                          radius * 2 + 1, radius * 2 + 1))
        ids = []
        seen = set()
        for i in range(radius):
            subr = id[radius-i:radius+i+1, radius-i:radius+i+1]
            subr_ids = set(list(np.unique(subr)))
            ids.extend(list(subr_ids - seen))
            seen |= subr_ids
        visuals = [VisualNode._visual_ids.get(x, None) for x in ids]
        return [v for v in visuals if v is not None]