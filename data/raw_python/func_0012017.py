def shutdown(self):
        """Executed on shutdown of application"""
        self.stopped.set()

        if hasattr(self.api, "shutdown"):
            self.api.shutdown()

        for thread in self.thread.values():
            thread.join()