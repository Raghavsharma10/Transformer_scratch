def stop(self):
        """Hard stop the server and sub process"""
        self._end.value = True
        if self.background_process:
            try:
                self.background_process.terminate()
            except Exception:
                pass
        for task_id, values in self.current_tasks.items():
            try:
                values['proc'].terminate()
            except Exception:
                pass