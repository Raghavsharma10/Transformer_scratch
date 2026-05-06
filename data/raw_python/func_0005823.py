def set_thread_params(
            self, enable=None, count=None, count_offload=None, stack_size=None, no_wait=None):
        """Sets threads related params.

        :param bool enable: Enable threads in the embedded languages.
            This will allow to spawn threads in your app.

            .. warning:: Threads will simply *not work* if this option is not enabled.
                There will likely be no error, just no execution of your thread code.

        :param int count: Run each worker in prethreaded mode with the specified number
            of threads per worker.

            .. warning:: Do not use with ``gevent``.

            .. note:: Enables threads automatically.

        :param int count_offload: Set the number of threads (per-worker) to spawn
            for offloading. Default: 0.

            These threads run such tasks in a non-blocking/evented way allowing
            for a huge amount of concurrency. Various components of the uWSGI stack
            are offload-friendly.

            .. note:: Try to set it to the number of CPU cores to take advantage of SMP.

            * http://uwsgi-docs.readthedocs.io/en/latest/OffloadSubsystem.html

        :param int stack_size: Set threads stacksize.

        :param bool no_wait: Do not wait for threads cancellation on quit/reload.

        """
        self._set('enable-threads', enable, cast=bool)
        self._set('no-threads-wait', no_wait, cast=bool)
        self._set('threads', count)
        self._set('offload-threads', count_offload)

        if count:
            self._section.print_out('Threads per worker: %s' % count)

        self._set('threads-stacksize', stack_size)

        return self._section