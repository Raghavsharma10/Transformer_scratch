def commit_config(self):
        """Commit configuration."""
        self.device.cu.commit(ignore_warning=self.ignore_warning)
        if not self.config_lock:
            self._unlock()