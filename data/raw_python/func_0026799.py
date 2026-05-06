def connect_controller(self, vid, pid, serial):
        """Connect a controller."""
        self.lib.tdConnectTellStickController(vid, pid, serial)