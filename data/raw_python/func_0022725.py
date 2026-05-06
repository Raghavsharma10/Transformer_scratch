def _update_line(self):
        """ Update border line to match new shape """
        w = self._border_width
        m = self.margin
        # border is drawn within the boundaries of the widget:
        #
        #  size = (8, 7)  margin=2
        #  internal rect = (3, 3, 2, 1)
        #  ........
        #  ........
        #  ..BBBB..
        #  ..B  B..
        #  ..BBBB..
        #  ........
        #  ........
        #
        l = b = m
        r = self.size[0] - m
        t = self.size[1] - m
        pos = np.array([
            [l, b], [l+w, b+w],
            [r, b], [r-w, b+w],
            [r, t], [r-w, t-w],
            [l, t], [l+w, t-w],
        ], dtype=np.float32)
        faces = np.array([
            [0, 2, 1],
            [1, 2, 3],
            [2, 4, 3],
            [3, 5, 4],
            [4, 5, 6],
            [5, 7, 6],
            [6, 0, 7],
            [7, 0, 1],
            [5, 3, 1],
            [1, 5, 7],
        ], dtype=np.int32)
        start = 8 if self._border_color.is_blank else 0
        stop = 8 if self._bgcolor.is_blank else 10
        face_colors = None
        if self._face_colors is not None:
            face_colors = self._face_colors[start:stop]
        self._mesh.set_data(vertices=pos, faces=faces[start:stop],
                            face_colors=face_colors)

        # picking mesh covers the entire area
        self._picking_mesh.set_data(vertices=pos[::2])