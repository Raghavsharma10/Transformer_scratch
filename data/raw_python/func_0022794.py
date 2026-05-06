def set_data(self, pos=None, symbol='o', size=10., edge_width=1.,
                 edge_width_rel=None, edge_color='black', face_color='white',
                 scaling=False):
        """ Set the data used to display this visual.

        Parameters
        ----------
        pos : array
            The array of locations to display each symbol.
        symbol : str
            The style of symbol to draw (see Notes).
        size : float or array
            The symbol size in px.
        edge_width : float | None
            The width of the symbol outline in pixels.
        edge_width_rel : float | None
            The width as a fraction of marker size. Exactly one of
            `edge_width` and `edge_width_rel` must be supplied.
        edge_color : Color | ColorArray
            The color used to draw each symbol outline.
        face_color : Color | ColorArray
            The color used to draw each symbol interior.
        scaling : bool
            If set to True, marker scales when rezooming.

        Notes
        -----
        Allowed style strings are: disc, arrow, ring, clobber, square, diamond,
        vbar, hbar, cross, tailed_arrow, x, triangle_up, triangle_down,
        and star.
        """
        assert (isinstance(pos, np.ndarray) and
                pos.ndim == 2 and pos.shape[1] in (2, 3))
        if (edge_width is not None) + (edge_width_rel is not None) != 1:
            raise ValueError('exactly one of edge_width and edge_width_rel '
                             'must be non-None')
        if edge_width is not None:
            if edge_width < 0:
                raise ValueError('edge_width cannot be negative')
        else:
            if edge_width_rel < 0:
                raise ValueError('edge_width_rel cannot be negative')
        self.symbol = symbol
        self.scaling = scaling

        edge_color = ColorArray(edge_color).rgba
        if len(edge_color) == 1:
            edge_color = edge_color[0]

        face_color = ColorArray(face_color).rgba
        if len(face_color) == 1:
            face_color = face_color[0]

        n = len(pos)
        data = np.zeros(n, dtype=[('a_position', np.float32, 3),
                                  ('a_fg_color', np.float32, 4),
                                  ('a_bg_color', np.float32, 4),
                                  ('a_size', np.float32, 1),
                                  ('a_edgewidth', np.float32, 1)])
        data['a_fg_color'] = edge_color
        data['a_bg_color'] = face_color
        if edge_width is not None:
            data['a_edgewidth'] = edge_width
        else:
            data['a_edgewidth'] = size*edge_width_rel
        data['a_position'][:, :pos.shape[1]] = pos
        data['a_size'] = size
        self.shared_program['u_antialias'] = self.antialias  # XXX make prop
        self._data = data
        self._vbo.set_data(data)
        self.shared_program.bind(self._vbo)
        self.update()