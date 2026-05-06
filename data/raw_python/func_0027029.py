def start(self, timeout=None, root_object=None):
        """ Starts listening to events.

            Args:
                timeout (int): number of seconds before timeout. Used for testing purpose only.
                root_object (bambou.NURESTRootObject): NURESTRootObject object that is listening. Used for testing purpose only.
        """

        if self._is_running:
            return

        if timeout:
            self._timeout = timeout
            self._start_time = int(time())

        pushcenter_logger.debug("[NURESTPushCenter] Starting push center on url %s ..." % self.url)
        self._is_running = True
        self.__root_object = root_object

        from .nurest_session import NURESTSession
        current_session = NURESTSession.get_current_session()
        args_session = {'session': current_session}

        self._thread = StoppableThread(target=self._listen, name='push-center', kwargs=args_session)
        self._thread.daemon = True
        self._thread.start()