def _run(self):
        """Run the iterative optimizer"""
        success = self.initialize()
        while success is None:
            success = self.propagate()
        return success