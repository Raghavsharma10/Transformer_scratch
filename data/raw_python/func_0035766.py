def list_overlay_names(self):
        """Return list of overlay names."""
        overlay_names = []
        if not os.path.isdir(self._overlays_abspath):
            return overlay_names
        for fname in os.listdir(self._overlays_abspath):
            name, ext = os.path.splitext(fname)
            overlay_names.append(name)
        return overlay_names