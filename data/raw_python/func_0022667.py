def _compute_iso_color(self):
        """ compute LineVisual color from level index and corresponding level
        color
        """
        level_color = []
        colors = self._lc
        for i, index in enumerate(self._li):
            level_color.append(np.zeros((index, 4)) + colors[i])
        self._cl = np.vstack(level_color)