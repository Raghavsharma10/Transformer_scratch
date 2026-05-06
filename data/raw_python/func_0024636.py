def load_roller_shutter(self, item):
        """Load roller shutter from JSON."""
        rollershutter = RollerShutter.from_config(self.pyvlx, item)
        self.add(rollershutter)