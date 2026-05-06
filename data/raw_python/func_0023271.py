def _update(self):
        """Rebuilds the shaders, and repositions the objects
           that are used internally by the ColorBarVisual
        """
        self._colorbar.halfdim = self._halfdim
        self._border.halfdim = self._halfdim

        self._label.text = self._label_str
        self._ticks[0].text = str(self._clim[0])
        self._ticks[1].text = str(self._clim[1])

        self._update_positions()

        self._colorbar._update()
        self._border._update()