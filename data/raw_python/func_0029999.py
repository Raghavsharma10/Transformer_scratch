def _run_events(self, tag, stage=None):
        """Run tests marked with a particular tag and stage"""

        self._run_event_methods(tag, stage)
        self._run_tests(tag, stage)