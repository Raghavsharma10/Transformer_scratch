def dpi(self):
        """ Physical resolution of the document coordinate system (dots per
        inch).
        """
        if self._dpi is None:
            if self._canvas is None:
                return None
            else:
                return self.canvas.dpi
        else:
            return self._dpi