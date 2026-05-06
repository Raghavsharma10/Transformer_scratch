def get_state(self):
        """Get general information about the state of the class"""
        return {"started": (True if self.background_process and
                            self.background_process.is_alive() else False),
                "paused": self._pause.value,
                "stopped": self._end.value,
                "tasks": len(self.current_tasks),
                "busy_tasks": len(self.busy_tasks),
                "free_tasks": len(self.free_tasks)}