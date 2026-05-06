def rgba(self, val):
        """Set the color using an Nx4 array of RGBA floats"""
        # Note: all other attribute sets get routed here!
        # This method is meant to do the heavy lifting of setting data
        rgba = _user_to_rgba(val, expand=False)
        if self._rgba is None:
            self._rgba = rgba  # only on init
        else:
            self._rgba[:, :rgba.shape[1]] = rgba