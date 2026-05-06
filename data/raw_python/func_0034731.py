def collect(self):
        """
        Create some concurrent workers that process the tasks simultaneously.
        """
        collected = super(Command, self).collect()
        if self.faster:
            self.worker_spawn_method()
            self.post_processor()
        return collected