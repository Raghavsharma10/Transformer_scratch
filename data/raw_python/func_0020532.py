def get_orchestrator_build_logs(self, build_id, follow=False, wait_if_missing=False):
        """
        provide logs from orchestrator build

        :param build_id: str
        :param follow: bool, fetch logs as they come?
        :param wait_if_missing: bool, if build doesn't exist, wait
        :return: generator yielding objects with attributes 'platform' and 'line'
        """
        logs = self.get_build_logs(build_id=build_id, follow=follow,
                                   wait_if_missing=wait_if_missing, decode=True)

        if logs is None:
            return
        if isinstance(logs, GeneratorType):
            for entries in logs:
                for entry in entries.splitlines():
                    yield LogEntry(*self._parse_build_log_entry(entry))
        else:
            for entry in logs.splitlines():
                yield LogEntry(*self._parse_build_log_entry(entry))