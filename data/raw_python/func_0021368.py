def serve_forever(self, fork=False):
		"""
		Start handling requests. This method must be called and does not
		return unless the :py:meth:`.shutdown` method is called from
		another thread.

		:param bool fork: Whether to fork or not before serving content.
		:return: The child processes PID if *fork* is set to True.
		:rtype: int
		"""
		if fork:
			if not hasattr(os, 'fork'):
				raise OSError('os.fork is not available')
			child_pid = os.fork()
			if child_pid != 0:
				self.logger.info('forked child process: ' + str(child_pid))
				return child_pid
		self.__server_thread = threading.current_thread()
		self.__wakeup_fd = WakeupFd()
		self.__is_shutdown.clear()
		self.__should_stop.clear()
		self.__is_running.set()
		while not self.__should_stop.is_set():
			try:
				self._serve_ready()
			except socket.error:
				self.logger.warning('encountered socket error, stopping server')
				self.__should_stop.set()
		self.__is_shutdown.set()
		self.__is_running.clear()
		return 0