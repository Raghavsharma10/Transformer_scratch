def disconnect_controller(self, vid, pid, serial):
        """Disconnect a controller."""
        self.lib.tdDisconnectTellStickController(vid, pid, serial)