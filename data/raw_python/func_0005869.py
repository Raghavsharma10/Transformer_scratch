def set_master_logging_params(
            self, enable=None, dedicate_thread=None, buffer=None,
            sock_stream=None, sock_stream_requests_only=None):
        """Sets logging params for delegating logging to master process.

        :param bool enable: Delegate logging to master process.
            Delegate the write of the logs to the master process
            (this will put all of the logging I/O to a single process).
            Useful for system with advanced I/O schedulers/elevators.

        :param bool dedicate_thread: Delegate log writing to a thread.

            As error situations could cause the master to block while writing
            a log line to a remote server, it may be a good idea to use this option and delegate
            writes to a secondary thread.

        :param int buffer: Set the buffer size for the master logger in bytes.
            Bigger log messages will be truncated.

        :param bool|tuple sock_stream: Create the master logpipe as SOCK_STREAM.

        :param bool|tuple sock_stream_requests_only: Create the master requests logpipe as SOCK_STREAM.

        """
        self._set('log-master', enable, cast=bool)
        self._set('threaded-logger', dedicate_thread, cast=bool)
        self._set('log-master-bufsize', buffer)

        self._set('log-master-stream', sock_stream, cast=bool)

        if sock_stream_requests_only:
            self._set('log-master-req-stream', sock_stream_requests_only, cast=bool)

        return self._section