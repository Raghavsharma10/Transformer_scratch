def get_build_logs(self, build_id, follow=False, build_json=None, wait_if_missing=False,
                       decode=False):
        """
        provide logs from build

        NOTE: Since atomic-reactor 1.6.25, logs are always in UTF-8, so if
        asked to decode, we assume that is the encoding in use. Otherwise, we
        return the bytes exactly as they came from the container.

        :param build_id: str
        :param follow: bool, fetch logs as they come?
        :param build_json: dict, to save one get-build query
        :param wait_if_missing: bool, if build doesn't exist, wait
        :param decode: bool, whether or not to decode logs as utf-8
        :return: None, bytes, or iterable of bytes
        """
        logs = self.os.logs(build_id, follow=follow, build_json=build_json,
                            wait_if_missing=wait_if_missing)

        if decode and isinstance(logs, GeneratorType):
            return self._decode_build_logs_generator(logs)

        # str or None returned from self.os.logs()
        if decode and logs is not None:
            logs = logs.decode("utf-8").rstrip()

        return logs