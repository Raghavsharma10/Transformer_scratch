def load_window_opener(self, item):
        """Load window opener from JSON."""
        window = Window.from_config(self.pyvlx, item)
        self.add(window)