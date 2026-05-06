def on_go(self, target):
        """
        RUN target WHEN SIGNALED
        """
        if not target:
            Log.error("expecting target")

        with self.lock:
            if not self._go:
                DEBUG and self._name and Log.note("Adding target to signal {{name|quote}}", name=self.name)

                if not self.job_queue:
                    self.job_queue = [target]
                else:
                    self.job_queue.append(target)
                return

        (DEBUG_SIGNAL) and Log.note("Signal {{name|quote}} already triggered, running job immediately", name=self.name)
        target()