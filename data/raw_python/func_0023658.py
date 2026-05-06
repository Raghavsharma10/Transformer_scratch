def start_process(self, key):
        """Start a specific processes."""
        if key in self.processes and key in self.paused:
            os.killpg(os.getpgid(self.processes[key].pid), signal.SIGCONT)
            self.queue[key]['status'] = 'running'
            self.paused.remove(key)
            return True
        elif key not in self.processes:
            if self.queue[key]['status'] in ['queued', 'stashed']:
                self.spawn_new(key)
                return True

        return False