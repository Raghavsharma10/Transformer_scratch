def attach(self, canvas):
        """Attach this tranform to a canvas

        Parameters
        ----------
        canvas : instance of Canvas
            The canvas.
        """
        self._canvas = canvas
        canvas.events.resize.connect(self.on_resize)
        canvas.events.mouse_wheel.connect(self.on_mouse_wheel)
        canvas.events.mouse_move.connect(self.on_mouse_move)