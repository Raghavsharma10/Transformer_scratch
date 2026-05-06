def redraw(self):
        """
        Redraw the Vispy canvas
        """
        if self._multiscat is not None:
            self._multiscat._update()
        self.vispy_widget.canvas.update()