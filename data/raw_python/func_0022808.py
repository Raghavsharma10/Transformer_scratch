def value(self, val):
        """Set the color using length-N array of (from HSV)"""
        hsv = self._hsv
        hsv[:, 2] = _array_clip_val(val)
        self.rgba = _hsv_to_rgb(hsv)