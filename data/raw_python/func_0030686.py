def start(self, phase, stage, **kwargs):
        """Start a new routine, stage or phase"""
        return ProgressSection(self, self._session, phase, stage, self._logger, **kwargs)