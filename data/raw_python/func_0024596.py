def start(self):
        """Create loop task."""
        self.run_task = self.pyvlx.loop.create_task(
            self.loop())