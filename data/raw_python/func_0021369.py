def shutdown(self):
		"""Shutdown the server and stop responding to requests."""
		self.__should_stop.set()
		if self.__server_thread == threading.current_thread():
			self.__is_shutdown.set()
			self.__is_running.clear()
		else:
			if self.__wakeup_fd is not None:
				os.write(self.__wakeup_fd.write_fd, b'\x00')
			self.__is_shutdown.wait()
		if self.__wakeup_fd is not None:
			self.__wakeup_fd.close()
			self.__wakeup_fd = None
		for server in self.sub_servers:
			server.shutdown()