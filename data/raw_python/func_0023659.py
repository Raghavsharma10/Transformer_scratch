def pause_process(self, key):
        """Pause a specific processes."""
        if key in self.processes and key not in self.paused:
            os.killpg(os.getpgid(self.processes[key].pid), signal.SIGSTOP)
            self.queue[key]['status'] = 'paused'
            self.paused.append(key)
            return True
        return False