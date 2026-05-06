def _update_positions(self):
        """
        updates the positions of the colorbars and labels

        """
        self._colorbar.pos = self._pos
        self._border.pos = self._pos

        if self._orientation == "right" or self._orientation == "left":
            self._label.rotation = -90

        x, y = self._pos
        halfw, halfh = self._halfdim

        label_anchors = \
            ColorBarVisual._get_label_anchors(center=self._pos,
                                              halfdim=self._halfdim,
                                              orientation=self._orientation,
                                              transforms=self.label.transforms)
        self._label.anchors = label_anchors

        ticks_anchors = \
            ColorBarVisual._get_ticks_anchors(center=self._pos,
                                              halfdim=self._halfdim,
                                              orientation=self._orientation,
                                              transforms=self.label.transforms)

        self._ticks[0].anchors = ticks_anchors
        self._ticks[1].anchors = ticks_anchors

        (label_pos, ticks_pos) = \
            ColorBarVisual._calc_positions(center=self._pos,
                                           halfdim=self._halfdim,
                                           border_width=self.border_width,
                                           orientation=self._orientation,
                                           transforms=self.transforms)

        self._label.pos = label_pos
        self._ticks[0].pos = ticks_pos[0]
        self._ticks[1].pos = ticks_pos[1]