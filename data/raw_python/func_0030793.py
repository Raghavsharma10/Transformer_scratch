def start_heartbeat(self):
        """ Reset hearbeat timer """
        self.stop_heartbeat()

        self._heartbeat_timer = task.LoopingCall(self._heartbeat)
        self._heartbeat_timer.start(self._heartbeat_interval, False)