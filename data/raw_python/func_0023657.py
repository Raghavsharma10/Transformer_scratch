def kill_all(self, kill_signal, kill_shell=False):
        """Kill all running processes."""
        for key in self.processes.keys():
            self.kill_process(key, kill_signal, kill_shell)