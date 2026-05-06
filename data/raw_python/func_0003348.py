async def with_exception(self, subprocess, *matchers):
        """
        Monitoring event matchers while executing a subprocess. If events are matched before the subprocess ends,
        the subprocess is terminated and a RoutineException is raised.
        """
        def _callback(event, matcher):
            raise RoutineException(matcher, event)
        return await self.with_callback(subprocess, _callback, *matchers)