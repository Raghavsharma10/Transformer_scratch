def visual_at(self, pos):
        """Return the visual at a given position

        Parameters
        ----------
        pos : tuple
            The position in logical coordinates to query.

        Returns
        -------
        visual : instance of Visual | None
            The visual at the position, if it exists.
        """
        tr = self.transforms.get_transform('canvas', 'framebuffer')
        fbpos = tr.map(pos)[:2]

        try:
            id_ = self._render_picking(region=(fbpos[0], fbpos[1],
                                               1, 1))
            vis = VisualNode._visual_ids.get(id_[0, 0], None)
        except RuntimeError:
            # Don't have read_pixels() support for IPython. Fall back to
            # bounds checking.
            return self._visual_bounds_at(pos)
        return vis